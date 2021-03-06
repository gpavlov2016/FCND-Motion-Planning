import argparse
import time
import msgpack
from enum import Enum, auto

import numpy as np

from planning_utils import *
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local
import csv

class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()


class MotionPlanning(Drone):

    def __init__(self, connection):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 1.0:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)


    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Searching for a path ...")
        TARGET_ALTITUDE = 5
        SAFETY_DISTANCE = 5

        self.target_position[2] = TARGET_ALTITUDE

        # read lat0, lon0 from colliders into floating point values
        filename = 'colliders.csv'
        f = open(filename, 'r')
        csv_reader = csv.reader(f)
        first_line = next(csv_reader)
        print(first_line)
        lat0 = first_line[0].split()[1]
        lon0 = first_line[1].split()[1]
        print(lat0, lon0)

        # set home position to (lon0, lat0, 0)
        self.set_home_position(lon0, lat0, 0)
        # retrieve current global position
        print(self.global_position)
        # convert to current local position using global_to_local()
        (self._north, self._east, self._down) = global_to_local(self.global_position, self.global_home)
        print('global home {0}, position {1}, local position {2}'.format(self.global_home, self.global_position,
                                                                         self.local_position))
        # Read in obstacle map
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)

        # Define a grid for a particular altitude and safety margin around obstacles
        #grid, north_offset, east_offset = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        # This is now the routine using Voronoi
        grid, edges, north_offset, east_offset = create_grid_and_edges(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        print('Grid size = ', (len(grid), len(grid[0])))
        print('Number of edges = ', len(edges))
        print('north_offset, east_offset = ', north_offset, east_offset)

        # Convert local position of the drone (can be negtive) to grid indexes (positive integers)
        grid_start = (self._north - north_offset, self._east - east_offset)
        print('grid_start = ', grid_start)
        # Set goal in Geodetic coordinates (lat, lon)
        goal_global = (-122.398986, 37.794309, 0) #275 Sacramento St
        # Convert goal from global to local coords
        goal_local = global_to_local(goal_global, self.global_home)
        # Convert to grid coordinates (integers)
        grid_goal = (goal_local[0] - north_offset, goal_local[1] - east_offset)
        print('grid_goal = ', grid_goal)

        print('Generating free space graph')
        g = nx.Graph()
        for e in edges:
            p1 = e[0]
            p2 = e[1]
            dist = LA.norm(np.array(p2) - np.array(p1))
            g.add_edge(p1, p2, weight=dist)

        '''
        #self.show_grid(grid, data, grid_start, grid_goal)
        nodes, polygons = generate_nodes(data, n_samples=300)
        import time
        t0 = time.time()
        print('Generating free space graph')
        g = create_graph(nodes, polygons, 10)
        print('graph took {0} seconds to build'.format(time.time() - t0))
        '''

        print("Number of nodes = ", len(g.nodes))
        print("Number of edges", len(g.edges))

        #show_graph(g, grid, g.nodes, data)

        #start = closest_point(g, (grid_start[0], grid_start[1], 0))
        #goal = closest_point(g, (grid_goal[0], grid_goal[1], 0))
        start = closest_point(g, grid_start)
        goal = closest_point(g, grid_goal)
        print('Graph start and goal: ', start, goal)

        # Run A* to find a path from start to goal
        print('Searching path')
        path, _ = a_star_graph(g, heuristic, start, goal)

        #show_path(g, grid, path, data, start, goal)

        print('Path length = ', len(path))

        if len(path) == 0:
            print('Path not found, termintating')
            exit(2)

        path = prune_path(path, grid)
        print('pruned path length', len(path))

        # Path is expressed in grid coordinates, shift it back to local coordinates
        waypoints = [[int(p[0] + north_offset), int(p[1] + east_offset), TARGET_ALTITUDE, 0] for p in path]

        # Set self.waypoints
        self.waypoints = waypoints
        # send waypoints to sim (this is just for visualization of waypoints)
        self.send_waypoints()

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
    drone = MotionPlanning(conn)
    time.sleep(1)

    drone.start()
