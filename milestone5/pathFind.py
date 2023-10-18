import pygame
import math
from queue import PriorityQueue
import json
import numpy as np
import time

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165 ,0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)

class Spot:
	def __init__(self, row, col, width, total_rows):
		self.row = row
		self.col = col
		self.x = row * width
		self.y = col * width
		self.color = WHITE
		self.neighbors = []
		self.width = width
		self.total_rows = total_rows

	def get_pos(self):
		return self.row, self.col

	def is_closed(self):
		return self.color == RED

	def is_open(self):
		return self.color == GREEN

	def is_barrier(self):
		return self.color == BLACK

	def is_start(self):
		return self.color == ORANGE

	def is_end(self):
		return self.color == TURQUOISE

	def reset(self):
		self.color = WHITE

	def make_start(self):
		self.color = ORANGE

	def make_closed(self):
		self.color = RED

	def make_open(self):
		self.color = GREEN

	def make_barrier(self):
		self.color = BLACK

	def make_end(self):
		self.color = TURQUOISE

	def make_path(self):
		self.color = PURPLE
        

	def update_neighbors(self, grid):
		self.neighbors = []
		if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier(): # DOWN
			self.neighbors.append(grid[self.row + 1][self.col])

		if self.row > 0 and not grid[self.row - 1][self.col].is_barrier(): # UP
			self.neighbors.append(grid[self.row - 1][self.col])

		if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier(): # RIGHT
			self.neighbors.append(grid[self.row][self.col + 1])

		if self.col > 0 and not grid[self.row][self.col - 1].is_barrier(): # LEFT
			self.neighbors.append(grid[self.row][self.col - 1])

		# Addition of corners
		# if self.row > 0 and self.col > 0 and not grid[self.row - 1][self.col - 1].is_barrier(): # UP LEFT
			# self.neighbors.append(grid[self.row - 1][self.col - 1])

		# if self.row > 0 and self.col < self.total_rows - 1 and not grid[self.row - 1][self.col + 1].is_barrier(): # UP RIGHT
			# self.neighbors.append(grid[self.row - 1][self.col + 1])

		# if self.row < self.total_rows - 1 and self.col < self.total_rows - 1 and not grid[self.row + 1][self.col + 1].is_barrier(): # DOWN RIGHT
			# self.neighbors.append(grid[self.row + 1][self.col + 1])

		# if self.row < self.total_rows - 1 and self.col > 0 and not grid[self.row + 1][self.col - 1].is_barrier(): # DOWN LEFT
			# self.neighbors.append(grid[self.row + 1][self.col - 1])



	def __lt__(self, other):
		return False


def h(p1, p2): # MODIFIED TO EUCLIDEAN DISTANCE
	x1, y1 = p1
	x2, y2 = p2
	return abs(x1 - x2) + abs(y1 - y2)


def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        current = came_from[current]
        
        # MODIFICATION TO EXTRACT COORDINATES
        row, col = current.get_pos()
        x, y = grid_to_coord(row, col, 33)
        path.append([x, y])
        current.make_path()
    return path


def algorithm(grid, start, end):
	count = 0
	open_set = PriorityQueue()
	open_set.put((0, count, start))
	came_from = {}
	g_score = {spot: float("inf") for row in grid for spot in row}
	g_score[start] = 0
	f_score = {spot: float("inf") for row in grid for spot in row}
	f_score[start] = h(start.get_pos(), end.get_pos())

	open_set_hash = {start}

	while not open_set.empty():

		current = open_set.get()[2]
		open_set_hash.remove(current)

		if current == end:
			path = reconstruct_path(came_from, end)
			end.make_end()
			return path

		for neighbor in current.neighbors:
			temp_g_score = g_score[current] + h(neighbor.get_pos(), current.get_pos())

			if temp_g_score < g_score[neighbor]:
				came_from[neighbor] = current
				g_score[neighbor] = temp_g_score
				f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
				if neighbor not in open_set_hash:
					count += 1
					open_set.put((f_score[neighbor], count, neighbor))
					open_set_hash.add(neighbor)
					neighbor.make_open()

		if current != start:
			current.make_closed()

	return False


def make_grid(rows, width):
	grid = []
	gap = width // rows
	for i in range(rows):
		grid.append([])
		for j in range(rows):
			spot = Spot(i, j, gap, rows)
			grid[i].append(spot)

	return grid

def groundtruth_to_grid(x,y,rows): # Convert ground truth coordinate to grid
    row = (x + 1.6)/round(3.2/rows,1)
    col = (1.6 - y)/round(3.2/rows,1)
    row = round(row)
    col = round(col)
    return row, col

def read_groundtruth(grid,rows):
    # READ GROUND TRUTH
    f = open("lab_output\M5_true_map.txt", "r")
    data = json.loads(f.read())
    for i in data:
        [row, col] = groundtruth_to_grid(data[i]['x'], data[i]['y'], rows)
        try:
            spot = grid[row][col]
            spot.make_barrier()
        except:
            pass
        try:
            spot = grid[row + 1][col]
            spot.make_barrier()
        except:
            pass
        try:
            spot = grid[row][col + 1]
            spot.make_barrier()
        except:
            pass
        try:
            spot = grid[row - 1][col]
            spot.make_barrier()
        except:
            pass
        try:
            spot = grid[row][col - 1]
            spot.make_barrier()
        except:
            pass
        try:
            spot = grid[row + 1][col + 1]
            spot.make_barrier()
        except:
            pass
        try:
            spot = grid[row + 1][col - 1]
            spot.make_barrier()
        except:
            pass
        try:
            spot = grid[row - 1][col + 1]
            spot.make_barrier()
        except:
            pass
        try:
            spot = grid[row - 1][col - 1]
            spot.make_barrier()
        except:
            pass
    f.close()

def add_obstacle(grid, rows, coord):
    x = coord[0]
    y = coord[1]
    if x < -1.6:
        x = -1.6
    if x > 1.6:
        x = 1.6
    if y < -1.6:
        y = -1.6
    if y > 1.6:
        y = 1.6
    [row, col] = groundtruth_to_grid(x, y, rows)
    try:
        spot = grid[row][col]
        spot.make_barrier()
    except:
        pass
    try:
        spot = grid[row + 1][col]
        spot.make_barrier()
    except:
        pass
    try:
        spot = grid[row][col + 1]
        spot.make_barrier()
    except:
        pass
    try:
        spot = grid[row - 1][col]
        spot.make_barrier()
    except:
        pass
    try:
        spot = grid[row][col - 1]
        spot.make_barrier()
    except:
        pass
    try:
        spot = grid[row + 1][col + 1]
        spot.make_barrier()
    except:
        pass
    try:
        spot = grid[row + 1][col - 1]
        spot.make_barrier()
    except:
        pass
    try:
        spot = grid[row - 1][col + 1]
        spot.make_barrier()
    except:
        pass
    try:
        spot = grid[row - 1][col - 1]
        spot.make_barrier()
    except:
        pass
    
def add_boundary(grid, rows):
    for x in range(-16, 17, 1):
        for y in range(-16, 17, 1):
            if x < -12 or x > 12 or y < -12 or y > 12:
                [row, col] = groundtruth_to_grid(round(x/10,1),round(y/10,1),rows)
                spot = grid[row][col]
                spot.make_barrier()
                
def grid_to_coord(row,col,rows): # Grid to coordinate
	x = round(3.2/rows,1)*row - 1.6
	y = 1.6 - round(3.2/rows,1)*col 
	return round(x,1),round(y,1)

def read_waypoint():
	x = []
	y = []
	f = open("waypoint.txt","r")
	data = json.loads(f.read())
	for i in data:
		x.append(data[i]['x'])
		y.append(data[i]['y'])
	f.close()
	return x,y

def simplify_path(path, threshold):
    # Path: list
    # threshold: distance in (m)

    new_path = []
    for j in range(len(path)-1):
        if j == 0:
            new_path.append(path[j])
        else:
            new_path.append(path[j])
            # Check same x coord
            if (new_path[-1][0] == new_path[-2][0]):
                if round(abs(new_path[-1][1] - new_path[-2][1]),2) < threshold:
                    if path[j+1][0] == new_path[-1][0]:
                        new_path.pop()
            # Check same y coord
            elif (new_path[-1][1] == new_path[-2][1]):
                if round(abs(new_path[-1][0] - new_path[-2][0]),2) < threshold:
                    if path[j+1][1] == new_path[-1][1]:
                        new_path.pop()
            # Check diagonal (same direction)
            elif (round(path[j+1][0] - path[j][0],2) == round(path[j][0] - path[j-1][0],2) and round(path[j+1][1] - path[j][1],2) == round(path[j][1] - path[j-1][1],2)):
                if h(new_path[-1],new_path[-2]) < threshold:
                    new_path.pop()
    new_path.append(path[-1])
    
    return new_path

# START: list [x1, y1]
# END: list [x2, y2]
# EXTRA: nested list [[x, y], [x, y]]
def main(START, END, EXTRA):
    ROWS = 33
    width = 40*ROWS
    grid = make_grid(ROWS, width)

    start = None
    end = None

    read_groundtruth(grid, ROWS)
    add_boundary(grid, ROWS)
    
    # CHECK IF ANY EXTRA FRUIT
    if len(EXTRA) != 0:
        for fruit in EXTRA:
            add_obstacle(grid, ROWS, [fruit[0],fruit[1]])

    # ADD START POINT
    row, col = groundtruth_to_grid(START[0], START[1], ROWS)
    if row >= ROWS:
        row = row - 1
    if col >= ROWS:
        col = col - 1
    start = grid[row][col]
    start.make_start()

    # ADD END POINT
    row, col = groundtruth_to_grid(END[0], END[1], ROWS)
    if row >= ROWS:
        row = row - 1
    if col >= ROWS:
        col = col - 1
    end = grid[row][col]
    if end.is_barrier():
        print("Obstacle in waypoint")
        return [], 100
    else:
        end.make_end()

    for row in grid:
        for spot in row:
            spot.update_neighbors(grid)

    path = algorithm(grid, start, end)

    # Reverse path
    try:
        path = path[::-1]
    except:
        return 0
    # Add end point
    path.append(END)

    # Simplify straight path
    # Longest segment = 0.4m

    threshold = 0.4
    path = simplify_path(path,threshold)

    turn = 0
    vertical = 0
    horizontal = 0
    for i in range(len(path)-1):
        if i == 0:
            if path[i][0] == path[i+1][0]:
                horizontal = 1
            else:
                vertical = 1
        else:
            if path[i][0] != path[i+1][0] and horizontal == 1:
                horizontal = 0
                vertical = 1
                turn = turn + 1
            elif path[i][1] != path[i+1][1] and vertical == 1:
                horizontal = 1
                vertical = 0
                turn = turn + 1
    
    return path, turn


# START = [0, 0]
# END = [-1.2, 0]
# EXTRA = []
# path, turn = main(START, END, EXTRA)
# print(path)
# print(turn)