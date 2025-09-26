"""
MIT License

Copyright (c) 2018 zhm-real
Copyright (c) 2023 zhangjiedev

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Reeds Shepp Path Planning Algorithm
Original source: https://github.com/zhm-real/MotionPlanning
This version has been adopted for the repository: https://github.com/zhangjiedev/Reeds-shepp-parking
"""

import math
import numpy as np
import matplotlib.pyplot as plt

import draw

STEP_SIZE = 0.5
MAX_LENGTH = 1000.0
PI = math.pi


# Class for PATH element
class PATH:

    def __init__(self, lengths, ctypes, L, x, y, yaw, directions):
        self.lengths = lengths
        self.ctypes = ctypes
        self.L = L
        self.x = x
        self.y = y
        self.yaw = yaw
        self.directions = directions


def calc_optimal_path(sx, sy, syaw, gx, gy, gyaw, maxc, step_size=STEP_SIZE):
    """
    Calculate the optimal Reed-Shepp path from a start to a goal state.
    """
    paths = calc_all_paths(sx,
                           sy,
                           syaw,
                           gx,
                           gy,
                           gyaw,
                           maxc,
                           step_size=step_size)

    if not paths:
        return None

    minL = paths[0].L
    mini = 0
    for i in range(len(paths)):
        if paths[i].L <= minL:
            minL, mini = paths[i].L, i

    return paths[mini]


def calc_all_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size=STEP_SIZE):
    q0 = [sx, sy, syaw]
    q1 = [gx, gy, gyaw]

    paths = generate_path(q0, q1, maxc)
    if not paths:
        return []

    for path in paths:
        x, y, yaw, directions = generate_local_course(path.L, path.lengths,
                                                      path.ctypes, maxc,
                                                      step_size * maxc)

        # Convert to global coordinate frame
        path.x = [
            math.cos(-q0[2]) * ix + math.sin(-q0[2]) * iy + q0[0]
            for (ix, iy) in zip(x, y)
        ]
        path.y = [
            -math.sin(-q0[2]) * ix + math.cos(-q0[2]) * iy + q0[1]
            for (ix, iy) in zip(x, y)
        ]
        path.yaw = [pi_2_pi(iyaw + q0[2]) for iyaw in yaw]
        path.directions = directions
        path.lengths = [l / maxc for l in path.lengths]
        path.L = path.L / maxc

    return paths


def set_path(paths, lengths, ctypes):
    path = PATH([], [], 0.0, [], [], [], [])
    path.ctypes = ctypes
    path.lengths = lengths

    # Check if a similar path already exists
    for path_e in paths:
        if path_e.ctypes == path.ctypes and sum(
                abs(x - y)
                for x, y in zip(path_e.lengths, path.lengths)) <= 0.01:
            return paths

    path.L = sum(abs(i) for i in lengths)
    if path.L >= MAX_LENGTH:
        return paths

    if path.L >= 0.01:
        paths.append(path)
    return paths


def LSL(x, y, phi):
    u, t = R(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if t >= 0.0:
        v = M(phi - t)
        if v >= 0.0:
            return True, t, u, v
    return False, 0.0, 0.0, 0.0


def LSR(x, y, phi):
    u1, t1 = R(x + math.sin(phi), y - 1.0 - math.cos(phi))
    u1 = u1**2
    if u1 >= 4.0:
        u = math.sqrt(u1 - 4.0)
        theta = math.atan2(2.0, u)
        t = M(t1 + theta)
        v = M(t - phi)
        if t >= 0.0 and v >= 0.0:
            return True, t, u, v
    return False, 0.0, 0.0, 0.0


def LRL(x, y, phi):
    u1, t1 = R(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if u1 <= 4.0:
        u = -2.0 * math.asin(0.25 * u1)
        t = M(t1 + 0.5 * u + PI)
        v = M(phi - t + u)
        if t >= 0.0 and u <= 0.0:
            return True, t, u, v
    return False, 0.0, 0.0, 0.0


def SLS(x, y, phi):
    phi = M(phi)
    if y > 0.0 and 0.0 < phi < PI * 0.99:
        xd = -y / math.tan(phi) + x
        t = xd - math.tan(phi / 2.0)
        u = phi
        v = math.sqrt((x - xd)**2 + y**2) - math.tan(phi / 2.0)
        return True, t, u, v
    elif y < 0.0 and 0.0 < phi < PI * 0.99:
        xd = -y / math.tan(phi) + x
        t = xd - math.tan(phi / 2.0)
        u = phi
        v = -math.sqrt((x - xd)**2 + y**2) - math.tan(phi / 2.0)
        return True, t, u, v
    return False, 0.0, 0.0, 0.0


def SCS(x, y, phi, paths):
    flag, t, u, v = SLS(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["S", "L", "S"])
    flag, t, u, v = SLS(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["S", "R", "S"])
    return paths


def CSC(x, y, phi, paths):
    flag, t, u, v = LSL(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["L", "S", "L"])
    flag, t, u, v = LSL(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["L", "S", "L"])
    flag, t, u, v = LSL(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["R", "S", "R"])
    flag, t, u, v = LSL(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["R", "S", "R"])
    flag, t, u, v = LSR(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["L", "S", "R"])
    flag, t, u, v = LSR(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["L", "S", "R"])
    flag, t, u, v = LSR(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["R", "S", "L"])
    flag, t, u, v = LSR(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["R", "S", "L"])
    return paths


def CCC(x, y, phi, paths):
    flag, t, u, v = LRL(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["L", "R", "L"])
    flag, t, u, v = LRL(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["L", "R", "L"])
    flag, t, u, v = LRL(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["R", "L", "R"])
    flag, t, u, v = LRL(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["R", "L", "R"])

    xb = x * math.cos(phi) + y * math.sin(phi)
    yb = x * math.sin(phi) - y * math.cos(phi)
    flag, t, u, v = LRL(xb, yb, phi)
    if flag:
        paths = set_path(paths, [v, u, t], ["L", "R", "L"])
    flag, t, u, v = LRL(-xb, yb, -phi)
    if flag:
        paths = set_path(paths, [-v, -u, -t], ["L", "R", "L"])
    flag, t, u, v = LRL(xb, -yb, -phi)
    if flag:
        paths = set_path(paths, [v, u, t], ["R", "L", "R"])
    flag, t, u, v = LRL(-xb, -yb, phi)
    if flag:
        paths = set_path(paths, [-v, -u, -t], ["R", "L", "R"])
    return paths


def generate_local_course(L, lengths, mode, maxc, step_size):
    point_num = int(L / step_size) + len(lengths) + 3
    px, py, pyaw, directions = [0.0] * point_num, [0.0] * point_num, [
        0.0
    ] * point_num, [0.0] * point_num
    ind = 1

    if lengths[0] > 0.0:
        directions[0] = 1
    else:
        directions[0] = -1

    d = step_size if lengths[0] > 0.0 else -step_size
    pd, ll = d, 0.0

    for m, l, i in zip(mode, lengths, range(len(mode))):
        d = step_size if l > 0.0 else -step_size
        ox, oy, oyaw = px[ind], py[ind], pyaw[ind]

        ind -= 1
        pd = d - ll if i >= 1 and (lengths[i - 1] * lengths[i]) <= 0 else d - ll

        while abs(pd) <= abs(l):
            ind += 1
            px, py, pyaw, directions = interpolate(ind, pd, m, maxc, ox, oy,
                                                   oyaw, px, py, pyaw,
                                                   directions)
            pd += d

        ll = l - pd + d
        ind += 1
        px, py, pyaw, directions = interpolate(ind, l, m, maxc, ox, oy, oyaw,
                                               px, py, pyaw, directions)

    # Remove unused data
    while px[-1] == 0.0:
        px.pop(), py.pop(), pyaw.pop(), directions.pop()
    return px, py, pyaw, directions


def interpolate(ind, l, m, maxc, ox, oy, oyaw, px, py, pyaw, directions):
    if m == "S":
        px[ind] = ox + l / maxc * math.cos(oyaw)
        py[ind] = oy + l / maxc * math.sin(oyaw)
        pyaw[ind] = oyaw
    else:
        ldx = math.sin(l) / maxc
        ldy = (1.0 - math.cos(l)) / (maxc if m == "L" else -maxc)
        gdx = math.cos(-oyaw) * ldx + math.sin(-oyaw) * ldy
        gdy = -math.sin(-oyaw) * ldx + math.cos(-oyaw) * ldy
        px[ind] = ox + gdx
        py[ind] = oy + gdy

    if m == "L":
        pyaw[ind] = oyaw + l
    elif m == "R":
        pyaw[ind] = oyaw - l

    directions[ind] = 1 if l > 0.0 else -1
    return px, py, pyaw, directions


def generate_path(q0, q1, maxc):
    dx = q1[0] - q0[0]
    dy = q1[1] - q0[1]
    dth = q1[2] - q0[2]
    c, s = math.cos(q0[2]), math.sin(q0[2])
    x = (c * dx + s * dy) * maxc
    y = (-s * dx + c * dy) * maxc

    paths = []
    paths = SCS(x, y, dth, paths)
    paths = CSC(x, y, dth, paths)
    paths = CCC(x, y, dth, paths)
    return paths


# Utility functions
def pi_2_pi(theta):
    while theta > PI:
        theta -= 2.0 * PI
    while theta < -PI:
        theta += 2.0 * PI
    return theta


def R(x, y):
    return math.hypot(x, y), math.atan2(y, x)


def M(theta):
    phi = theta % (2.0 * PI)
    if phi < -PI:
        phi += 2.0 * PI
    if phi > PI:
        phi -= 2.0 * PI
    return phi


if __name__ == '__main__':
    main()
