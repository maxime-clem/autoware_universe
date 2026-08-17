/**
 * Shared host/device geometry for analytic path-tracking costs (polyline distance, OBB overlap).
 */
#pragma once

#ifndef MPPI_COST_FUNCTIONS_PATH_TRACKING_GEOMETRY_CUH_
#define MPPI_COST_FUNCTIONS_PATH_TRACKING_GEOMETRY_CUH_

#include <algorithm>
#include <cmath>

namespace mppi
{
namespace cost
{
namespace detail
{
#ifdef __CUDA_ARCH__
__device__ inline float clampUnitInterval(const float t)
{
  return fmaxf(0.0F, fminf(1.0F, t));
}

__device__ inline float vectorLength(const float dx, const float dy)
{
  return sqrtf(dx * dx + dy * dy);
}
#else
inline float clampUnitInterval(const float t)
{
  return std::max(0.0F, std::min(1.0F, t));
}

inline float vectorLength(const float dx, const float dy)
{
  return std::sqrt(dx * dx + dy * dy);
}
#endif

__host__ __device__ inline float distancePointToSegment(
  const float px, const float py, const float x0, const float y0, const float x1, const float y1)
{
  const float dx = x1 - x0;
  const float dy = y1 - y0;
  const float len_sq = dx * dx + dy * dy;
  if (len_sq < 1.0E-8F) {
    return vectorLength(px - x0, py - y0);
  }

  const float t = clampUnitInterval(((px - x0) * dx + (py - y0) * dy) / len_sq);
  return vectorLength(px - (x0 + t * dx), py - (y0 + t * dy));
}

/** Perpendicular distance to the infinite line through (x0,y0)-(x1,y1). */
__host__ __device__ inline float perpendicularDistanceToLine(
  const float px, const float py, const float x0, const float y0, const float x1, const float y1)
{
  const float dx = x1 - x0;
  const float dy = y1 - y0;
  const float len_sq = dx * dx + dy * dy;
  if (len_sq < 1.0E-8F) {
    return vectorLength(px - x0, py - y0);
  }
#ifdef __CUDA_ARCH__
  return fabsf((px - x0) * dy - (py - y0) * dx) / sqrtf(len_sq);
#else
  return std::fabs((px - x0) * dy - (py - y0) * dx) / std::sqrt(len_sq);
#endif
}

/**
 * Cross-track distance to a polyline. Interior projections use point-to-segment
 * distance; projections past the first/last vertex use perpendicular distance to
 * the extended terminal segment (so overshooting the horizon is not treated as
 * lateral departure).
 */
struct PolylineProjection
{
  float lateral_distance = 0.0F;
  int best_i = 0;
  /** Unclamped segment parameter from the closest-segment search. */
  float best_t_raw = 0.5F;
};

/** Distance and unclamped parameter of point (px,py) onto segment i. */
__host__ __device__ inline void distanceToPolylineSegment(
  const float px, const float py, const float * poly_x, const float * poly_y, const int i,
  float & dist, float & t_raw)
{
  const float x0 = poly_x[i];
  const float y0 = poly_y[i];
  const float x1 = poly_x[i + 1];
  const float y1 = poly_y[i + 1];
  const float dx = x1 - x0;
  const float dy = y1 - y0;
  const float len_sq = dx * dx + dy * dy;
  if (len_sq < 1.0E-8F) {
    dist = vectorLength(px - x0, py - y0);
    t_raw = 0.0F;
    return;
  }
  t_raw = ((px - x0) * dx + (py - y0) * dy) / len_sq;
  const float t = clampUnitInterval(t_raw);
  dist = vectorLength(px - (x0 + t * dx), py - (y0 + t * dy));
}

__host__ __device__ inline void finalizePolylineProjection(
  PolylineProjection & result, const float px, const float py, const float * poly_x,
  const float * poly_y, const int n_pts, const int best_i, const float best_t_raw,
  const float best_dist)
{
  result.best_i = best_i;
  result.best_t_raw = best_t_raw;
  // Past the start/end of the finite polyline: use true cross-track to the extended tip.
  if (best_i == 0 && best_t_raw < 0.0F) {
    result.lateral_distance =
      perpendicularDistanceToLine(px, py, poly_x[0], poly_y[0], poly_x[1], poly_y[1]);
    return;
  }
  if (best_i == n_pts - 2 && best_t_raw > 1.0F) {
    result.lateral_distance = perpendicularDistanceToLine(
      px, py, poly_x[n_pts - 2], poly_y[n_pts - 2], poly_x[n_pts - 1], poly_y[n_pts - 1]);
    return;
  }
  result.lateral_distance = best_dist;
}

/**
 * Closest-segment projection onto a polyline.
 * @param hint_i warm-start segment index from a previous projection (e.g. last
 *               timestep). When >= 0, performs a local hill-climb from that
 *               segment (amortized O(1) along a trajectory). When < 0, full
 *               O(n) scan — use for the first query or when no prior exists.
 */
__host__ __device__ inline PolylineProjection projectPointToPolyline(
  const float px, const float py, const float * poly_x, const float * poly_y, const int n_pts,
  const int hint_i = -1)
{
  PolylineProjection result;
  if (n_pts <= 0) {
    return result;
  }
  if (n_pts == 1) {
    result.lateral_distance = vectorLength(px - poly_x[0], py - poly_y[0]);
    result.best_i = 0;
    result.best_t_raw = 0.0F;
    return result;
  }

  const int n_seg = n_pts - 1;
  float best_dist = 1.0E8F;
  int best_i = 0;
  float best_t_raw = 0.5F;

  if (hint_i < 0) {
    for (int i = 0; i < n_seg; ++i) {
      float dist = 0.0F;
      float t_raw = 0.0F;
      distanceToPolylineSegment(px, py, poly_x, poly_y, i, dist, t_raw);
      if (dist < best_dist) {
        best_dist = dist;
        best_i = i;
        best_t_raw = t_raw;
      }
    }
  } else {
    // Local search from previous solution (Newton-style warm start).
    best_i = hint_i;
    if (best_i >= n_seg) {
      best_i = n_seg - 1;
    }
    distanceToPolylineSegment(px, py, poly_x, poly_y, best_i, best_dist, best_t_raw);
    for (int iter = 0; iter < n_seg; ++iter) {
      float left_dist = best_dist;
      float left_t = best_t_raw;
      float right_dist = best_dist;
      float right_t = best_t_raw;
      if (best_i > 0) {
        distanceToPolylineSegment(px, py, poly_x, poly_y, best_i - 1, left_dist, left_t);
      }
      if (best_i < n_seg - 1) {
        distanceToPolylineSegment(px, py, poly_x, poly_y, best_i + 1, right_dist, right_t);
      }
      if (left_dist < best_dist && left_dist <= right_dist) {
        --best_i;
        best_dist = left_dist;
        best_t_raw = left_t;
        continue;
      }
      if (right_dist < best_dist) {
        ++best_i;
        best_dist = right_dist;
        best_t_raw = right_t;
        continue;
      }
      break;
    }
  }

  finalizePolylineProjection(result, px, py, poly_x, poly_y, n_pts, best_i, best_t_raw, best_dist);
  return result;
}

__host__ __device__ inline float crossTrackDistanceToPolyline(
  const float px, const float py, const float * poly_x, const float * poly_y, const int n_pts)
{
  return projectPointToPolyline(px, py, poly_x, poly_y, n_pts).lateral_distance;
}

/**
 * Arc length along a polyline at the projected footpoint.
 * @param total_s full polyline length (final cumulative chord length).
 * Tip overhang clamps path length to [0, total_s]; remaining = total_s - path_length.
 * When past the tip (last segment, t_raw > 1), overshoot_distance_s is the along-track
 * extension beyond the terminal vertex; otherwise 0.
 */
__host__ __device__ inline void pathLengthAtProjection(
  const PolylineProjection & proj, const float * poly_x, const float * poly_y,
  const float * cumulative_s, const int n_pts, const float total_s, float & path_length_s,
  float & remaining_distance_s, float & overshoot_distance_s)
{
  path_length_s = 0.0F;
  remaining_distance_s = total_s;
  overshoot_distance_s = 0.0F;
  if (n_pts <= 0) {
    return;
  }
  if (n_pts == 1) {
    path_length_s = total_s;
    remaining_distance_s = 0.0F;
    return;
  }

  if (proj.best_i == 0 && proj.best_t_raw < 0.0F) {
    path_length_s = 0.0F;
    remaining_distance_s = total_s;
    return;
  }
  if (proj.best_i == n_pts - 2 && proj.best_t_raw > 1.0F) {
    path_length_s = total_s;
    remaining_distance_s = 0.0F;
    overshoot_distance_s =
      (proj.best_t_raw - 1.0F) *
      vectorLength(poly_x[n_pts - 1] - poly_x[n_pts - 2], poly_y[n_pts - 1] - poly_y[n_pts - 2]);
    return;
  }

  const float t = clampUnitInterval(proj.best_t_raw);
  if (cumulative_s != nullptr) {
    const float s0 = cumulative_s[proj.best_i];
    const float s1 = cumulative_s[proj.best_i + 1];
    path_length_s = s0 + t * (s1 - s0);
  } else {
    float s = 0.0F;
    for (int i = 0; i < proj.best_i; ++i) {
      s += vectorLength(poly_x[i + 1] - poly_x[i], poly_y[i + 1] - poly_y[i]);
    }
    path_length_s = s + t * vectorLength(
                              poly_x[proj.best_i + 1] - poly_x[proj.best_i],
                              poly_y[proj.best_i + 1] - poly_y[proj.best_i]);
  }
  remaining_distance_s = total_s - path_length_s;
}

/** Signed lateral offset from segment; positive = left of forward tangent. */
__host__ __device__ inline float signedLateralOffsetPointToSegment(
  const float px, const float py, const float x0, const float y0, const float x1, const float y1)
{
  const float dx = x1 - x0;
  const float dy = y1 - y0;
  const float len_sq = dx * dx + dy * dy;
  if (len_sq < 1.0E-8F) {
    return px - x0;
  }

  const float t = clampUnitInterval(((px - x0) * dx + (py - y0) * dy) / len_sq);
  const float cx = x0 + t * dx;
  const float cy = y0 + t * dy;
  const float len = vectorLength(dx, dy);
  return ((px - cx) * (-dy) + (py - cy) * dx) / len;
}

__host__ __device__ inline float dot2(
  const float ax, const float ay, const float bx, const float by)
{
  return ax * bx + ay * by;
}

/** Separating-axis test for two oriented boxes (body x = forward, y = left). */
__host__ __device__ inline bool orientedBoxesOverlap(
  const float acx, const float acy, const float a_cos, const float a_sin, const float a_hl,
  const float a_hw, const float bcx, const float bcy, const float b_cos, const float b_sin,
  const float b_hl, const float b_hw)
{
  const float tx = bcx - acx;
  const float ty = bcy - acy;

  const float a0x = a_cos;
  const float a0y = a_sin;
  const float a1x = -a_sin;
  const float a1y = a_cos;
  const float b0x = b_cos;
  const float b0y = b_sin;
  const float b1x = -b_sin;
  const float b1y = b_cos;

#ifdef __CUDA_ARCH__
#define MPPI_COST_ABS(x) fabsf(x)
#else
#define MPPI_COST_ABS(x) std::fabs(x)
#endif

  float t = MPPI_COST_ABS(dot2(tx, ty, a0x, a0y));
  float r = a_hl + MPPI_COST_ABS(dot2(b0x, b0y, a0x, a0y)) * b_hl +
            MPPI_COST_ABS(dot2(b1x, b1y, a0x, a0y)) * b_hw;
  if (t > r) {
    return false;
  }

  t = MPPI_COST_ABS(dot2(tx, ty, a1x, a1y));
  r = a_hw + MPPI_COST_ABS(dot2(b0x, b0y, a1x, a1y)) * b_hl +
      MPPI_COST_ABS(dot2(b1x, b1y, a1x, a1y)) * b_hw;
  if (t > r) {
    return false;
  }

  t = MPPI_COST_ABS(dot2(tx, ty, b0x, b0y));
  r = b_hl + MPPI_COST_ABS(dot2(a0x, a0y, b0x, b0y)) * a_hl +
      MPPI_COST_ABS(dot2(a1x, a1y, b0x, b0y)) * a_hw;
  if (t > r) {
    return false;
  }

  t = MPPI_COST_ABS(dot2(tx, ty, b1x, b1y));
  r = b_hw + MPPI_COST_ABS(dot2(a0x, a0y, b1x, b1y)) * a_hl +
      MPPI_COST_ABS(dot2(a1x, a1y, b1x, b1y)) * a_hw;
  if (t > r) {
    return false;
  }

#undef MPPI_COST_ABS
  return true;
}

/** Four corners in body frame order: front-left, front-right, rear-right, rear-left. */
__host__ __device__ inline void orientedBoxCorners(
  const float cx, const float cy, const float cos_yaw, const float sin_yaw, const float half_length,
  const float half_width, float corners_x[4], float corners_y[4])
{
  const float fx = cos_yaw;
  const float fy = sin_yaw;
  const float lx = -sin_yaw;
  const float ly = cos_yaw;

  corners_x[0] = cx + half_length * fx + half_width * lx;
  corners_y[0] = cy + half_length * fy + half_width * ly;
  corners_x[1] = cx + half_length * fx - half_width * lx;
  corners_y[1] = cy + half_length * fy - half_width * ly;
  corners_x[2] = cx - half_length * fx - half_width * lx;
  corners_y[2] = cy - half_length * fy - half_width * ly;
  corners_x[3] = cx - half_length * fx + half_width * lx;
  corners_y[3] = cy - half_length * fy + half_width * ly;
}

/** Ray-casting point-in-polygon (closed boundary; vertices in order). */
__host__ __device__ inline bool pointInPolygon(
  const float px, const float py, const float * vertices_x, const float * vertices_y,
  const int vertex_count)
{
  if (vertex_count < 3) {
    return false;
  }

  bool inside = false;
  int j = vertex_count - 1;
  for (int i = 0; i < vertex_count; ++i) {
    const float xi = vertices_x[i];
    const float yi = vertices_y[i];
    const float xj = vertices_x[j];
    const float yj = vertices_y[j];
    const bool intersects =
      ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi + 1.0E-12F) + xi);
    if (intersects) {
      inside = !inside;
    }
    j = i;
  }
  return inside;
}
}  // namespace detail
}  // namespace cost
}  // namespace mppi

#endif  // MPPI_COST_FUNCTIONS_PATH_TRACKING_GEOMETRY_CUH_
