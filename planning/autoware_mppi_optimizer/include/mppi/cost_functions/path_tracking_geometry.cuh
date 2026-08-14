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

__host__ __device__ inline float cross2(
  const float ax, const float ay, const float bx, const float by)
{
  return ax * by - ay * bx;
}

/** Euclidean distance between two finite line segments (zero when they intersect). */
__host__ __device__ inline float distanceSegmentToSegment(
  const float ax0, const float ay0, const float ax1, const float ay1, const float bx0,
  const float by0, const float bx1, const float by1)
{
  const float adx = ax1 - ax0;
  const float ady = ay1 - ay0;
  const float bdx = bx1 - bx0;
  const float bdy = by1 - by0;
  const float denominator = cross2(adx, ady, bdx, bdy);
  const float qpx = bx0 - ax0;
  const float qpy = by0 - ay0;

  if (fabsf(denominator) > 1.0E-8F) {
    const float t = cross2(qpx, qpy, bdx, bdy) / denominator;
    const float u = cross2(qpx, qpy, adx, ady) / denominator;
    if (t >= 0.0F && t <= 1.0F && u >= 0.0F && u <= 1.0F) {
      return 0.0F;
    }
  }

  const float d0 = distancePointToSegment(ax0, ay0, bx0, by0, bx1, by1);
  const float d1 = distancePointToSegment(ax1, ay1, bx0, by0, bx1, by1);
  const float d2 = distancePointToSegment(bx0, by0, ax0, ay0, ax1, ay1);
  const float d3 = distancePointToSegment(bx1, by1, ax0, ay0, ax1, ay1);
  return fminf(fminf(d0, d1), fminf(d2, d3));
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
__host__ __device__ inline float crossTrackDistanceToPolyline(
  const float px, const float py, const float * poly_x, const float * poly_y, const int n_pts)
{
  if (n_pts <= 0) {
    return 0.0F;
  }
  if (n_pts == 1) {
    return vectorLength(px - poly_x[0], py - poly_y[0]);
  }

  float best_dist = 1.0E8F;
  int best_i = 0;
  float best_t_raw = 0.5F;
  for (int i = 0; i < n_pts - 1; ++i) {
    const float x0 = poly_x[i];
    const float y0 = poly_y[i];
    const float x1 = poly_x[i + 1];
    const float y1 = poly_y[i + 1];
    const float dx = x1 - x0;
    const float dy = y1 - y0;
    const float len_sq = dx * dx + dy * dy;
    float t_raw = 0.0F;
    float dist = 0.0F;
    if (len_sq < 1.0E-8F) {
      dist = vectorLength(px - x0, py - y0);
      t_raw = 0.0F;
    } else {
      t_raw = ((px - x0) * dx + (py - y0) * dy) / len_sq;
      const float t = clampUnitInterval(t_raw);
      dist = vectorLength(px - (x0 + t * dx), py - (y0 + t * dy));
    }
    if (dist < best_dist) {
      best_dist = dist;
      best_i = i;
      best_t_raw = t_raw;
    }
  }

  // Past the start/end of the finite polyline: use true cross-track to the extended tip.
  if (best_i == 0 && best_t_raw < 0.0F) {
    return perpendicularDistanceToLine(px, py, poly_x[0], poly_y[0], poly_x[1], poly_y[1]);
  }
  if (best_i == n_pts - 2 && best_t_raw > 1.0F) {
    return perpendicularDistanceToLine(
      px, py, poly_x[n_pts - 2], poly_y[n_pts - 2], poly_x[n_pts - 1], poly_y[n_pts - 1]);
  }
  return best_dist;
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

/** Signed OBB separation: positive clearance, zero contact, negative SAT penetration depth. */
__host__ __device__ inline float signedDistanceBetweenOrientedBoxes(
  const float acx, const float acy, const float a_cos, const float a_sin, const float a_hl,
  const float a_hw, const float bcx, const float bcy, const float b_cos, const float b_sin,
  const float b_hl, const float b_hw)
{
  if (!orientedBoxesOverlap(
        acx, acy, a_cos, a_sin, a_hl, a_hw, bcx, bcy, b_cos, b_sin, b_hl, b_hw)) {
    float ax[4];
    float ay[4];
    float bx[4];
    float by[4];
    orientedBoxCorners(acx, acy, a_cos, a_sin, a_hl, a_hw, ax, ay);
    orientedBoxCorners(bcx, bcy, b_cos, b_sin, b_hl, b_hw, bx, by);
    float min_distance = 1.0E8F;
#pragma unroll
    for (int ai = 0; ai < 4; ++ai) {
      const int an = (ai + 1) & 3;
#pragma unroll
      for (int bi = 0; bi < 4; ++bi) {
        const int bn = (bi + 1) & 3;
        min_distance = fminf(
          min_distance,
          distanceSegmentToSegment(ax[ai], ay[ai], ax[an], ay[an], bx[bi], by[bi], bx[bn], by[bn]));
      }
    }
    return min_distance;
  }

  const float tx = bcx - acx;
  const float ty = bcy - acy;
  const float axes_x[4] = {a_cos, -a_sin, b_cos, -b_sin};
  const float axes_y[4] = {a_sin, a_cos, b_sin, b_cos};
  float min_overlap = 1.0E8F;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float axis_x = axes_x[i];
    const float axis_y = axes_y[i];
    const float center_distance = fabsf(dot2(tx, ty, axis_x, axis_y));
    const float a_radius = a_hl * fabsf(dot2(a_cos, a_sin, axis_x, axis_y)) +
                           a_hw * fabsf(dot2(-a_sin, a_cos, axis_x, axis_y));
    const float b_radius = b_hl * fabsf(dot2(b_cos, b_sin, axis_x, axis_y)) +
                           b_hw * fabsf(dot2(-b_sin, b_cos, axis_x, axis_y));
    min_overlap = fminf(min_overlap, a_radius + b_radius - center_distance);
  }
  return -min_overlap;
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
