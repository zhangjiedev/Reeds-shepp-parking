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

Hybrid A* Path Planning Algorithm
Original source: https://github.com/zhm-real/MotionPlanning
This version has been adopted for the repository: https://github.com/zhangjiedev/Reeds-shepp-parking
"""

import os
import sys
import math
import heapq
from heapdict import heapdict
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.kdtree as kd
import configparser

import astar as astar
import draw as draw
import reeds_shepp as rs


# --- START OF CONFIG LOADING CLASS ---
class Config:

    def __init__(self, filename='settings.ini'):
        config = configparser.ConfigParser()
        config.read(filename)

        def get_float(section, key):
            return config.getfloat(section, key)

        def get_int(section, key):
            return config.getint(section, key)

        # Constants
        self.PI = get_float('Constants', 'PI')
        self.XY_RESO = get_float('Constants', 'XY_RESO')
        self.YAW_RESO = math.radians(get_float(
            'Constants', 'YAW_RESO_DEG'))  # Converted to radians
        self.MOVE_STEP = get_float('Constants', 'MOVE_STEP')
        self.N_STEER = get_float('Constants', 'N_STEER')
        self.COLLISION_CHECK_STEP = get_int('Constants', 'COLLISION_CHECK_STEP')
        self.EXTEND_BOUND = get_float('Constants', 'EXTEND_BOUND')

        # PathPlanning
        self.MAX_STEER = get_float('PathPlanning', 'MAX_STEER')

        # VehicleDimensions
        self.RF = get_float('VehicleDimensions', 'RF')
        self.RB = get_float('VehicleDimensions', 'RB')
        self.W = get_float('VehicleDimensions', 'W')
        self.WD = get_float('VehicleDimensions', 'WD')
        self.WB = get_float('VehicleDimensions', 'WB')
        self.TR = get_float('VehicleDimensions', 'TR')
        self.TW = get_float('VehicleDimensions', 'TW')

        # CostParameters
        self.GEAR_COST = get_float('CostParameters', 'GEAR_COST')
        self.BACKWARD_COST = get_float('CostParameters', 'BACKWARD_COST')
        self.STEER_CHANGE_COST = get_float('CostParameters',
                                           'STEER_CHANGE_COST')
        self.STEER_ANGLE_COST = get_float('CostParameters', 'STEER_ANGLE_COST')
        self.H_COST = get_float('CostParameters', 'H_COST')


# Instantiate the config object to be used throughout the module
C = Config()
# --- END OF CONFIG LOADING CLASS ---


class Node:

    def __init__(self, xind, yind, yawind, direction, x, y, yaw, directions,
                 steer, cost, pind):
        self.xind = xind
        self.yind = yind
        self.yawind = yawind
        self.direction = direction
        self.x = x
        self.y = y
        self.yaw = yaw
        self.directions = directions
        self.steer = steer
        self.cost = cost
        self.pind = pind


class Para:

    def __init__(self, minx, miny, minyaw, maxx, maxy, maxyaw, xw, yw, yaww,
                 xyreso, yawreso, ox, oy, kdtree):
        self.minx = minx
        self.miny = miny
        self.minyaw = minyaw
        self.maxx = maxx
        self.maxy = maxy
        self.maxyaw = maxyaw
        self.xw = xw
        self.yw = yw
        self.yaww = yaww
        self.xyreso = xyreso
        self.yawreso = yawreso
        self.ox = ox
        self.oy = oy
        self.kdtree = kdtree


class Path:

    def __init__(self, x, y, yaw, direction, cost):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.direction = direction
        self.cost = cost


class QueuePrior:

    def __init__(self):
        self.queue = heapdict()

    def empty(self):
        return len(self.queue) == 0  # if Q is empty

    def put(self, item, priority):
        self.queue[item] = priority  # push

    def get(self):
        return self.queue.popitem()[0]  # pop out element with smallest priority


def hybrid_astar_planning(sx, sy, syaw, gx, gy, gyaw, ox, oy, xyreso, yawreso):
    # Use config constant in the resolution calculation
    sxr, syr = round(sx / xyreso), round(sy / xyreso)
    gxr, gyr = round(gx / xyreso), round(gy / xyreso)
    # Use config constant in the resolution calculation
    syawr = round(rs.pi_2_pi(syaw) / yawreso)
    # Use config constant in the resolution calculation
    gyawr = round(rs.pi_2_pi(gyaw) / yawreso)

    nstart = Node(sxr, syr, syawr, 1, [sx], [sy], [syaw], [1], 0.0, 0.0, -1)
    ngoal = Node(gxr, gyr, gyawr, 1, [gx], [gy], [gyaw], [1], 0.0, 0.0, -1)

    kdtree = kd.KDTree([[x, y] for x, y in zip(ox, oy)])
    P = calc_parameters(ox, oy, xyreso, yawreso, kdtree)

    hmap = astar.calc_holonomic_heuristic_with_obstacle(ngoal, P.ox, P.oy,
                                                        P.xyreso, 1.0)
    steer_set, direc_set = calc_motion_set()
    open_set, closed_set = {calc_index(nstart, P): nstart}, {}

    qp = QueuePrior()
    qp.put(calc_index(nstart, P), calc_hybrid_cost(nstart, hmap, P))

    while True:
        if not open_set:
            return None

        ind = qp.get()
        n_curr = open_set[ind]
        closed_set[ind] = n_curr
        open_set.pop(ind)

        update, fpath = update_node_with_analystic_expantion(n_curr, ngoal, P)

        if update:
            fnode = fpath
            break

        for i in range(len(steer_set)):
            node = calc_next_node(n_curr, ind, steer_set[i], direc_set[i], P)

            if not node:
                continue

            node_ind = calc_index(node, P)

            if node_ind in closed_set:
                continue

            if node_ind not in open_set:
                open_set[node_ind] = node
                qp.put(node_ind, calc_hybrid_cost(node, hmap, P))
            else:
                if open_set[node_ind].cost > node.cost:
                    open_set[node_ind] = node
                    qp.put(node_ind, calc_hybrid_cost(node, hmap, P))

    return extract_path(closed_set, fnode, nstart)


def hybrid_astar_planning_k_paths(sx,
                                  sy,
                                  syaw,
                                  gx,
                                  gy,
                                  gyaw,
                                  ox,
                                  oy,
                                  xyreso,
                                  yawreso,
                                  k_max=3):
    """
    Finds up to k_max collision-free paths using Hybrid A*.

    :param k_max: The maximum number of paths to find (default is 3).
    :return: A list of Path objects, up to k_max in length.
    """
    sxr, syr = round(sx / xyreso), round(sy / xyreso)
    gxr, gyr = round(gx / xyreso), round(gy / xyreso)
    syawr = round(rs.pi_2_pi(syaw) / yawreso)
    gyawr = round(rs.pi_2_pi(gyaw) / yawreso)

    nstart = Node(sxr, syr, syawr, 1, [sx], [sy], [syaw], [1], 0.0, 0.0, -1)
    ngoal = Node(gxr, gyr, gyawr, 1, [gx], [gy], [gyaw], [1], 0.0, 0.0, -1)

    kdtree = kd.KDTree([[x, y] for x, y in zip(ox, oy)])
    P = calc_parameters(ox, oy, xyreso, yawreso, kdtree)

    hmap = astar.calc_holonomic_heuristic_with_obstacle(ngoal, P.ox, P.oy,
                                                        P.xyreso, 1.0)
    steer_set, direc_set = calc_motion_set()

    # open_set tracks the last known best node leading to a grid index
    open_set, closed_set = {calc_index(nstart, P): nstart}, {}

    qp = QueuePrior()
    qp.put(calc_index(nstart, P), calc_hybrid_cost(nstart, hmap, P))

    found_paths = []

    # This set will track the indices of the path-ending nodes that have
    # already successfully connected to the goal via analytic expansion,
    # to avoid redundant path extraction from the *same* grid cell.
    goal_parent_indices = set()

    while True:
        if not open_set and len(found_paths) < k_max:
            # Search failed or all nodes explored before finding k_max paths
            break

        if len(found_paths) >= k_max:
            # Found the required number of paths
            break

        ind = qp.get()
        n_curr = open_set[ind]

        # Move the node from open to closed set
        closed_set[ind] = n_curr
        open_set.pop(ind)

        # 1. Try analytic expansion (Reeds-Shepp) to the goal
        update, fpath = update_node_with_analystic_expantion(n_curr, ngoal, P)

        if update:
            # A potential path to the goal is found (P_k)
            fnode = fpath

            # Use the index of the parent node (n_curr) to track if we've
            # already found a path originating from this grid cell.
            if fnode.pind not in goal_parent_indices:
                path = extract_path(closed_set, fnode, nstart)
                found_paths.append(path)
                goal_parent_indices.add(fnode.pind)

            # Since we want k paths, DON'T break here, continue searching.
            # The A* structure ensures the paths are found in order of cost.
            if len(found_paths) >= k_max:
                break

        # 2. Continue discrete expansion (A* step)
        for i in range(len(steer_set)):
            node = calc_next_node(n_curr, ind, steer_set[i], direc_set[i], P)

            if not node:
                continue

            node_ind = calc_index(node, P)

            # Check if this state has already been closed with a better cost
            if node_ind in closed_set and closed_set[node_ind].cost <= node.cost:
                continue

            if node_ind not in open_set:
                open_set[node_ind] = node
                qp.put(node_ind, calc_hybrid_cost(node, hmap, P))
            else:
                if open_set[node_ind].cost > node.cost:
                    open_set[node_ind] = node
                    qp.put(node_ind, calc_hybrid_cost(node, hmap, P))

    return found_paths  # Returns a list of paths (up to k_max)


def extract_path(closed, ngoal, nstart):
    rx, ry, ryaw, direc = [], [], [], []
    cost = 0.0
    node = ngoal

    while True:
        rx += node.x[::-1]
        ry += node.y[::-1]
        ryaw += node.yaw[::-1]
        direc += node.directions[::-1]
        cost += node.cost

        if is_same_grid(node, nstart):
            break

        node = closed[node.pind]

    rx = rx[::-1]
    ry = ry[::-1]
    ryaw = ryaw[::-1]
    direc = direc[::-1]

    direc[0] = direc[1]
    path = Path(rx, ry, ryaw, direc, cost)

    return path


def calc_next_node(n_curr, c_id, u, d, P):
    # Use config constant
    step = C.XY_RESO * 2

    # Use config constant
    nlist = math.ceil(step / C.MOVE_STEP)
    # Use config constant
    xlist = [n_curr.x[-1] + d * C.MOVE_STEP * math.cos(n_curr.yaw[-1])]
    # Use config constant
    ylist = [n_curr.y[-1] + d * C.MOVE_STEP * math.sin(n_curr.yaw[-1])]
    # Use config constant
    yawlist = [
        rs.pi_2_pi(n_curr.yaw[-1] + d * C.MOVE_STEP / C.WB * math.tan(u))
    ]

    for i in range(nlist - 1):
        # Use config constant
        xlist.append(xlist[i] + d * C.MOVE_STEP * math.cos(yawlist[i]))
        # Use config constant
        ylist.append(ylist[i] + d * C.MOVE_STEP * math.sin(yawlist[i]))
        # Use config constant
        yawlist.append(
            rs.pi_2_pi(yawlist[i] + d * C.MOVE_STEP / C.WB * math.tan(u)))

    xind = round(xlist[-1] / P.xyreso)
    yind = round(ylist[-1] / P.xyreso)
    yawind = round(yawlist[-1] / P.yawreso)

    if not is_index_ok(xind, yind, xlist, ylist, yawlist, P):
        return None

    cost = 0.0

    if d > 0:
        direction = 1
        cost += abs(step)
    else:
        direction = -1
        # Use config constant
        cost += abs(step) * C.BACKWARD_COST

    # Use config constant
    if direction != n_curr.direction:  # switch back penalty
        cost += C.GEAR_COST

    # Use config constant
    cost += C.STEER_ANGLE_COST * abs(u)  # steer angle penalyty
    # Use config constant
    cost += C.STEER_CHANGE_COST * abs(n_curr.steer - u)  # steer change penalty
    cost = n_curr.cost + cost

    directions = [direction for _ in range(len(xlist))]

    node = Node(xind, yind, yawind, direction, xlist, ylist, yawlist,
                directions, u, cost, c_id)

    return node


def is_index_ok(xind, yind, xlist, ylist, yawlist, P):
    if xind <= P.minx or \
            xind >= P.maxx or \
            yind <= P.miny or \
            yind >= P.maxy:
        return False

    # Use config constant
    ind = range(0, len(xlist), C.COLLISION_CHECK_STEP)

    nodex = [xlist[k] for k in ind]
    nodey = [ylist[k] for k in ind]
    nodeyaw = [yawlist[k] for k in ind]

    if is_collision(nodex, nodey, nodeyaw, P):
        return False

    return True


def update_node_with_analystic_expantion(n_curr, ngoal, P):
    path = analystic_expantion(n_curr, ngoal, P)  # rs path: n -> ngoal

    if not path:
        return False, None

    fx = path.x[1:-1]
    fy = path.y[1:-1]
    fyaw = path.yaw[1:-1]
    fd = path.directions[1:-1]

    fcost = n_curr.cost + calc_rs_path_cost(path)
    fpind = calc_index(n_curr, P)
    fsteer = 0.0

    fpath = Node(n_curr.xind, n_curr.yind, n_curr.yawind, n_curr.direction, fx,
                 fy, fyaw, fd, fsteer, fcost, fpind)

    return True, fpath


def analystic_expantion(node, ngoal, P):
    sx, sy, syaw = node.x[-1], node.y[-1], node.yaw[-1]
    gx, gy, gyaw = ngoal.x[-1], ngoal.y[-1], ngoal.yaw[-1]

    # Use config constant
    maxc = math.tan(C.MAX_STEER) / C.WB
    # Use config constant
    paths = rs.calc_all_paths(sx,
                              sy,
                              syaw,
                              gx,
                              gy,
                              gyaw,
                              maxc,
                              step_size=C.MOVE_STEP)

    if not paths:
        return None

    pq = QueuePrior()
    for path in paths:
        pq.put(path, calc_rs_path_cost(path))

    while not pq.empty():
        path = pq.get()
        # Use config constant
        ind = range(0, len(path.x), C.COLLISION_CHECK_STEP)

        pathx = [path.x[k] for k in ind]
        pathy = [path.y[k] for k in ind]
        pathyaw = [path.yaw[k] for k in ind]

        if not is_collision(pathx, pathy, pathyaw, P):
            return path

    return None


def is_collision(x, y, yaw, P):
    for ix, iy, iyaw in zip(x, y, yaw):
        # Use config constant
        d = C.EXTEND_BOUND
        # Use config constant
        dl = (C.RF - C.RB) / 2.0
        # Use config constant
        r = (C.RF + C.RB) / 2.0 + d

        cx = ix + dl * math.cos(iyaw)
        cy = iy + dl * math.sin(iyaw)

        ids = P.kdtree.query_ball_point([cx, cy], r)

        if not ids:
            continue

        for i in ids:
            xo = P.ox[i] - cx
            yo = P.oy[i] - cy
            dx = xo * math.cos(iyaw) + yo * math.sin(iyaw)
            dy = -xo * math.sin(iyaw) + yo * math.cos(iyaw)

            # Use config constant
            if abs(dx) < r and abs(dy) < C.W / 2 + d:
                return True

    return False


def calc_rs_path_cost(rspath):
    cost = 0.0

    for lr in rspath.lengths:
        if lr >= 0:
            cost += abs(lr)  # Should use abs(lr) for length/distance traveled
        else:
            # Use config constant
            cost += abs(lr) * C.BACKWARD_COST

    for i in range(len(rspath.lengths) - 1):
        # Use config constant
        if rspath.lengths[i] * rspath.lengths[i + 1] < 0.0:
            cost += C.GEAR_COST

    for ctype in rspath.ctypes:
        if ctype != "S":
            # Use config constant
            cost += C.STEER_ANGLE_COST * abs(C.MAX_STEER)

    nctypes = len(rspath.ctypes)
    ulist = [0.0 for _ in range(nctypes)]

    for i in range(nctypes):
        if rspath.ctypes[i] == "R":
            # Use config constant
            ulist[i] = -C.MAX_STEER
        elif rspath.ctypes[
                i] == "L":  # Note: "L" is often used for Left in Reeds-Shepp
            # Use config constant
            ulist[i] = C.MAX_STEER

    for i in range(nctypes - 1):
        # Use config constant
        cost += C.STEER_CHANGE_COST * abs(ulist[i + 1] - ulist[i])

    return cost


def calc_hybrid_cost(node, hmap, P):
    # Use config constant
    cost = node.cost + \
           C.H_COST * hmap[node.xind - P.minx][node.yind - P.miny]

    return cost


def calc_motion_set():
    # Use config constants
    s = np.arange(C.MAX_STEER / C.N_STEER, C.MAX_STEER, C.MAX_STEER / C.N_STEER)

    steer = list(s) + [0.0] + list(-s)
    direc = [1.0 for _ in range(len(steer))] + [-1.0 for _ in range(len(steer))]
    steer = steer + steer

    return steer, direc


def is_same_grid(node1, node2):
    if node1.xind != node2.xind or \
            node1.yind != node2.yind or \
            node1.yawind != node2.yawind:
        return False

    return True


def calc_index(node, P):
    # UPDATED: Calculates a 1D index where the row index increases downwards

    # x index (column)
    col_index = node.xind - P.minx

    # y index (row) now increases downwards: 0 (top) to P.yw - 1 (bottom)
    # y_from_bottom_left is the original row index, increasing upwards
    y_from_bottom_left = node.yind - P.miny
    row_index_top_left = P.yw - 1 - y_from_bottom_left

    yaw_index = node.yawind - P.minyaw

    # The 1D index is yaw_index * grid_area + row_index * width + col_index
    ind = yaw_index * P.xw * P.yw + \
          row_index_top_left * P.xw + \
          col_index

    return ind


def calc_parameters(ox, oy, xyreso, yawreso, kdtree):
    minx = round(min(ox) / xyreso)
    miny = round(min(oy) / xyreso)
    maxx = round(max(ox) / xyreso)
    maxy = round(max(oy) / xyreso)

    xw, yw = maxx - minx, maxy - miny

    # Use config constant
    minyaw = round(-C.PI / yawreso) - 1
    # Use config constant
    maxyaw = round(C.PI / yawreso)
    yaww = maxyaw - minyaw

    return Para(minx, miny, minyaw, maxx, maxy, maxyaw, xw, yw, yaww, xyreso,
                yawreso, ox, oy, kdtree)


def draw_car(x, y, yaw, steer, color='black'):
    # Use config constants
    car = np.array([[-C.RB, -C.RB, C.RF, C.RF, -C.RB],
                    [C.W / 2, -C.W / 2, -C.W / 2, C.W / 2, C.W / 2]])

    # Use config constants
    wheel = np.array([[-C.TR, -C.TR, C.TR, C.TR, -C.TR],
                      [C.TW / 4, -C.TW / 4, -C.TW / 4, C.TW / 4, C.TW / 4]])

    rlWheel = wheel.copy()
    rrWheel = wheel.copy()
    frWheel = wheel.copy()
    flWheel = wheel.copy()

    Rot1 = np.array([[math.cos(yaw), -math.sin(yaw)],
                     [math.sin(yaw), math.cos(yaw)]])

    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    frWheel = np.dot(Rot2, frWheel)
    flWheel = np.dot(Rot2, flWheel)

    # Use config constants
    frWheel += np.array([[C.WB], [-C.WD / 2]])
    # Use config constants
    flWheel += np.array([[C.WB], [C.WD / 2]])
    # Use config constant
    rrWheel[1, :] -= C.WD / 2
    # Use config constant
    rlWheel[1, :] += C.WD / 2

    frWheel = np.dot(Rot1, frWheel)
    flWheel = np.dot(Rot1, flWheel)

    rrWheel = np.dot(Rot1, rrWheel)
    rlWheel = np.dot(Rot1, rlWheel)
    car = np.dot(Rot1, car)

    frWheel += np.array([[x], [y]])
    flWheel += np.array([[x], [y]])
    rrWheel += np.array([[x], [y]])
    rlWheel += np.array([[x], [y]])
    car += np.array([[x], [y]])

    plt.plot(car[0, :], car[1, :], color)
    plt.plot(frWheel[0, :], frWheel[1, :], color)
    plt.plot(rrWheel[0, :], rrWheel[1, :], color)
    plt.plot(flWheel[0, :], flWheel[1, :], color)
    plt.plot(rlWheel[0, :], rlWheel[1, :], color)
    # Use config constant
    draw.Arrow(x, y, yaw, C.WB * 0.8, color)


def design_obstacles(x, y):
    ox, oy = [], []

    for i in range(x):
        ox.append(i)
        oy.append(0)
    for i in range(x):
        ox.append(i)
        oy.append(y - 1)
    for i in range(y):
        ox.append(0)
        oy.append(i)
    for i in range(y):
        ox.append(x - 1)
        oy.append(i)
    for i in range(10, 21):
        ox.append(i)
        oy.append(15)
    for i in range(15):
        ox.append(20)
        oy.append(i)
    for i in range(15, 30):
        ox.append(30)
        oy.append(i)
    for i in range(16):
        ox.append(40)
        oy.append(i)

    return ox, oy


def main():
    print("start!")
    x, y = 51, 31
    sx, sy, syaw0 = 10.0, 7.0, np.deg2rad(120.0)
    gx, gy, gyaw0 = 45.0, 20.0, np.deg2rad(90.0)

    ox, oy = design_obstacles(x, y)

    t0 = time.time()
    # Use config constants
    path = hybrid_astar_planning(sx, sy, syaw0, gx, gy, gyaw0, ox, oy,
                                 C.XY_RESO, C.YAW_RESO)
    t1 = time.time()
    print("running T: ", t1 - t0)

    if not path:
        print("Searching failed!")
        return

    x = path.x
    y = path.y
    yaw = path.yaw
    direction = path.direction
    print(len(x))
    for k in range(len(x)):
        plt.cla()
        plt.plot(ox, oy, "sk")
        plt.plot(x, y, linewidth=1.5, color='r')

        if k < len(x) - 2:
            # Use config constant
            dy = (yaw[k + 1] - yaw[k]) / C.MOVE_STEP
            # Use config constant
            steer = rs.pi_2_pi(math.atan(-C.WB * dy / direction[k]))
        else:
            steer = 0.0

        draw_car(gx, gy, gyaw0, 0.0, 'dimgray')
        draw_car(x[k], y[k], yaw[k], steer)
        plt.title("Hybrid A*")
        plt.axis("equal")

        # UPDATED: Invert the y-axis to make the origin (0,0) appear at the top-left for visualization.
        plt.gca().invert_yaxis()

        plt.pause(0.0001)

    plt.show()
    print("Done!")


if __name__ == '__main__':
    main()
