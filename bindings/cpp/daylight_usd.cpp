// SPDX-License-Identifier: MIT
// USD reader: LightUSD composed stage (sublayers, references, payloads, ...) -> Mesh[] in world metres, Z-up.
#include "daylight_usd.h"
#pragma warning(push)
#pragma warning(disable : 4127)   // LightUSD headers: constant conditionals in templates
#include <composition.hh>
#include <core/model-scope.hh>
#include <io-util.hh>
#include <lightusd.hh>
#include <tydra/scene-access.hh>
#include <usdGeom.hh>
#pragma warning(pop)

namespace daylight {
namespace {

// visibility (default time) and purpose of an imageable prim of type T; false when the prim is not a T.
template <class T>
bool imageable_of(const lightusd::Prim& p, lightusd::Visibility& v, lightusd::Purpose& pu) {
  const T* t = p.as<T>();
  if (!t) return false;
  t->visibility.get_value().get(lightusd::value::TimeCode::Default(), &v);
  pu = t->purpose.get_value();
  return true;
}

bool prim_skipped(const lightusd::Prim& p) {
  // ponytail: instanceable prims and PointInstancers are skipped, not expanded; flatten / de-instance the stage first.
  if (!p.metas().get_active() || p.IsInstance() || p.is<lightusd::GeomPointInstancer>()) return true;
  lightusd::Visibility v = lightusd::Visibility::Inherited;
  lightusd::Purpose pu = lightusd::Purpose::Default;
  (void)(imageable_of<lightusd::Xform>(p, v, pu) || imageable_of<lightusd::Scope>(p, v, pu) ||
         imageable_of<lightusd::GeomMesh>(p, v, pu));
  return v == lightusd::Visibility::Invisible || pu == lightusd::Purpose::Proxy || pu == lightusd::Purpose::Guide;
}

// Row-vector convention: p' = [p,1] * W. Winding flipped when leftHanded XOR det(W3) < 0; authored normals ignored.
// Zero-copy reference to the default value of a typed attribute; time-sampled attributes fall back to a copy in tmp.
template <class T, class F>
const T& scalar_or(const lightusd::TypedAttribute<lightusd::Animatable<T>>& a, T& tmp, F fallback) {
  const auto& v = a.get_value_ref();
  return v && v->is_scalar() ? v->get_scalar_ref() : (tmp = fallback());
}

bool convert(const lightusd::GeomMesh& g, const lightusd::value::matrix4d& W, const std::string& name, bool yup,
             double mpu, std::vector<Mesh>& out, std::string& err) {
  std::vector<lightusd::value::point3f> ptsTmp;
  std::vector<int32_t> cntTmp, idxTmp;
  const auto& pts = scalar_or(g.points, ptsTmp, [&] { return g.get_points(); });
  const auto& cnt = scalar_or(g.faceVertexCounts, cntTmp, [&] { return g.get_faceVertexCounts(); });
  const auto& idx = scalar_or(g.faceVertexIndices, idxTmp, [&] { return g.get_faceVertexIndices(); });
  if (pts.empty() || cnt.empty()) return true;
  const double (*m)[4] = W.m;
  const double det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                     m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  Mesh mesh;
  mesh.name = name;
  mesh.points.resize(pts.size() * 3);
  for (size_t i = 0; i < pts.size(); ++i) {
    const auto& p = pts[i];
    double v[3];
    for (int j = 0; j < 3; ++j) v[j] = (p.x * m[0][j] + p.y * m[1][j] + p.z * m[2][j] + m[3][j]) * mpu;
    if (yup) { const double y = v[1]; v[1] = -v[2]; v[2] = y; }
    mesh.points[3 * i] = v[0]; mesh.points[3 * i + 1] = v[1]; mesh.points[3 * i + 2] = v[2];
  }
  mesh.counts.reserve(cnt.size()); mesh.indices.reserve(idx.size());
  std::vector<int32_t> holes;
  if (const auto h = g.holeIndices.get_value()) h->get(lightusd::value::TimeCode::Default(), &holes);
  std::vector<char> isHole(cnt.size(), 0);
  for (const int h : holes) if (h >= 0 && size_t(h) < cnt.size()) isHole[size_t(h)] = 1;
  // subdivisionScheme != none: the control cage is used as is.
  const bool flip = (g.orientation.get_value() == lightusd::Orientation::LeftHanded) != (det < 0);
  const int np = int(pts.size());
  size_t off = 0;
  for (size_t f = 0; f < cnt.size(); ++f) {
    const int n = cnt[f];
    if (n < 0 || off + size_t(n) > idx.size()) { err = name + ": faceVertexCounts/faceVertexIndices mismatch"; return false; }
    if (!isHole[f]) {
      mesh.counts.push_back(n);
      for (int k = 0; k < n; ++k) {
        const int i = idx[off + size_t(flip ? n - 1 - k : k)];
        if (i < 0 || i >= np) { err = name + ": faceVertexIndices out of range"; return false; }
        mesh.indices.push_back(i);
      }
    }
    off += size_t(n);
  }
  out.push_back(std::move(mesh));
  return true;
}

bool walk(const lightusd::tydra::XformNode& n, bool skip, bool yup, double mpu, std::vector<Mesh>& out, std::string& err) {
  if (n.prim) {
    skip = skip || prim_skipped(*n.prim);
    if (!skip)
      if (const auto* g = n.prim->as<lightusd::GeomMesh>())
        if (!convert(*g, n.get_world_matrix(), n.absolute_path.full_path_name(), yup, mpu, out, err)) return false;
  }
  for (const auto& c : n.children) if (!walk(c, skip, yup, mpu, out, err)) return false;
  return true;
}

}  // namespace

bool read_usd(const std::string& path, char upAxis, double metersPerUnit, std::vector<Mesh>& out, std::string& err) {
  std::string warn, lerr;
  lightusd::USDLoadOptions lopt;
  lopt.max_memory_limit_in_mb = 16384;   // LightUSD default 512 MB rejects 1M-face stages
  lightusd::Layer layer;
  if (!lightusd::LoadLayerFromFile(path, &layer, &warn, &lerr, lopt)) {   // UTF-8 paths: io-util.cc UTF8ToWchar
    err = path + ": " + lerr;
    return false;
  }
  std::string base = lightusd::io::GetBaseDir(path);
  if (base.empty()) base = ".";
  lightusd::AssetResolutionResolver resolver;
  resolver.set_current_working_path(base);
  resolver.set_search_paths({base});
  lightusd::SublayersCompositionOptions sopt;
  sopt.allow_parent_relative_paths = sopt.error_when_asset_not_found = true;   // LightUSD default: warn + drop
  lightusd::AllArcsCompositionOptions aopt;
  aopt.references.allow_parent_relative_paths = aopt.payload.allow_parent_relative_paths = true;
  aopt.references.error_when_asset_not_found = aopt.payload.error_when_asset_not_found = true;
  const std::string hint = ": composition failed, flatten the stage first (usdcat --flatten): ";
  // USD resolves stage metadata over the root layer stack; CompositeSublayers keeps only the root layer's metas.
  // ponytail: one level of sublayers (strongest first), nested subLayers of a sublayer are not visited.
  for (const auto& sl : layer.metas().subLayers) {
    lightusd::LayerMetas& rm = layer.metas();
    if (rm.upAxis.authored() && rm.metersPerUnit.authored()) break;
    lightusd::Layer sub;
    if (!lightusd::LoadLayerFromFile(resolver.resolve(sl.assetPath.GetAssetPath()), &sub, &warn, &lerr, lopt)) continue;
    if (!rm.upAxis.authored()) rm.upAxis = sub.metas().upAxis;
    if (!rm.metersPerUnit.authored()) rm.metersPerUnit = sub.metas().metersPerUnit;
  }
  if (!layer.metas().subLayers.empty()) {
    lightusd::Layer next;
    if (!lightusd::CompositeSublayers(resolver, layer, &next, &warn, &lerr, sopt)) { err = path + hint + lerr; return false; }
    layer = std::move(next);
  }
  for (int it = 0; layer.check_unresolved_references() || layer.check_unresolved_payload() || layer.check_unresolved_inherits() ||
                   layer.check_unresolved_variant() || layer.check_unresolved_specializes(); ++it) {
    lightusd::Layer next;
    if (it == 32 || !lightusd::CompositeAllArcs(resolver, layer, &next, &warn, &lerr, aopt)) {
      err = path + hint + (it == 32 ? "arcs still unresolved after 32 passes" : lerr);
      return false;
    }
    layer = std::move(next);
  }
  lightusd::Stage stage;
  if (!lightusd::LayerToStage(std::move(layer), &stage, &warn, &lerr)) { err = path + hint + lerr; return false; }
  const lightusd::StageMetas& sm = stage.metas();
  // USD spec fallbacks: metersPerUnit 0.01 (LightUSD's own is 1.0, hence authored()), upAxis Y. Arguments override.
  const double mpu = metersPerUnit > 0 ? metersPerUnit : sm.metersPerUnit.authored() ? sm.metersPerUnit.get_value() : 0.01;
  const bool yup = upAxis == 'Z' ? false : upAxis == 'Y' ? true : !(sm.upAxis.authored() && sm.upAxis.get_value() == lightusd::Axis::Z);
  lightusd::tydra::XformNode root;
  if (!lightusd::tydra::BuildXformNodeFromStage(stage, &root)) { err = path + ": cannot build the transform tree"; return false; }
  return walk(root, false, yup, mpu, out, err);
}

}  // namespace daylight
