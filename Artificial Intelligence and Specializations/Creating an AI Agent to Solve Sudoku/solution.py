from copy import copy
import random

def cross(A, B):
    "Cross product of elements in A and elements in B."
    boxes = [x + y for x in A for y in B]
    return boxes

def diagonal_unit_calculation(rows, cols):
    # Find all main diagonal units of the sudoku
    # Args:
    #   the row and column names of the sudoku as a string

    # Returns:
    #   The two main diagonal units of the sudoku as a list of lists

    diagonal_units = []
    temp_rows = rows
    for diagonals in range(2):
        temp = []
        if diagonals == 1:
            temp_rows = rows[::-1]
        for i in range(9):
            temp.append(temp_rows[i] + cols[i]) 
        diagonal_units.append(temp)
    return diagonal_units

# Useful variables
assignments = []
rows = 'ABCDEFGHI'
cols = '123456789'
boxes = cross(rows,cols)
row_units = [cross(x,cols) for x in rows]
cols_units = [cross(rows,x) for x in cols]
square_units = [cross(x,y) for x in ('ABC', 'DEF', 'GHI') for y in ('123', '456', '789')]
diagonal_units = diagonal_unit_calculation(rows, cols)
unit_list = row_units + cols_units + square_units + diagonal_units
units = dict((x, [y for y in unit_list if x in y]) for x in boxes)
peers = dict((x, set(sum(units[x], [])) - set([x])) for x in boxes)

def assign_value(values, box, value):
    """
    Please use this function to update your values dictionary!
    Assigns a value to a given box. If it updates the board record it.
    """

    # Don't waste memory appending actions that don't actually change any values
    if values[box] == value:
        return values

    values[box] = value
    if len(value) == 1:
        assignments.append(values.copy())
    return values

def naked_twins(values):
    """Eliminate values using the naked twins strategy.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        the values dictionary with the naked twins eliminated from peers.
    """

    # Find all boxes with only two values available
    possible_twins = [box for box in values if len(values[box]) == 2]
    for possible_twin in possible_twins:

        # For each of the candidate boxes find their twins (if they exist)
        twin_boxes = [peer for peer in peers[possible_twin] if values[peer] == values[possible_twin]]
        for twin_box in twin_boxes:
            
            # For each twin eliminate the respective value from the peers of their unit
            common_peers = peers[possible_twin] & peers[twin_box]
            for common_peer in common_peers:
                for number in values[possible_twin]:
                    if number in values[common_peer] and len(values[common_peer]) > 1:
                        values[common_peer] = values[common_peer].replace(number, '')
 
    return values

def grid_values(grid):
    """
    Convert grid into a dict of {square: char} with '123456789' for empties.
    Args:
        grid(string) - A grid in string form.
    Returns:
        A grid in dictionary form
            Keys: The boxes, e.g., 'A1'
            Values: The value in each box, e.g., '8'. If the box has no value, then the value will be '123456789'.
    """
    values = {}
    assert len(grid) == len(boxes)
    for i in range(len(boxes)):
        if grid[i] == '.':
            values[boxes[i]] = '123456789'
        else:
            values[boxes[i]] = grid[i]
    return values

def display(values):
    """
    Display the values as a 2-D grid.
    Input: The sudoku in dictionary form
    Output: None
    """
    width = 1 + max(len(values[s]) for s in boxes)
    line = '+'.join(['-'*(width*3)]*3)
    for r in rows:
        print(''.join(values[r+c].center(width)+('|' if c in '36' else '')
                      for c in cols))
        if r in 'CF': print(line)
    return

def eliminate(values):

    # Take the values from boxes that contain only one and eliminate them from their peers
    # Args:
    #     values(dict): The sudoku in dictionary form
    # Returns:
    #     values(dict): A reduced version of the sudoku

    single_digit_boxes = [box for box in values if len(values[box]) == 1]
    
    for box in single_digit_boxes:
        for peer in peers[box]:
            if values[box] in values[peer] and len(values[peer]) > 1:
                values[peer] = values[peer].replace(values[box], '')
    return values


def only_choice(values):

    # After elimination, go through the units of the boxes that contain possible values
    # and see whether some boxes contain values that are only choices
    # Args:
    #     values(dict): The sudoku in dictionary form
    # Returns:
    #     values(dict): A reduced version of the sudoku

    for unit in unit_list:
        for number in cols:
            boxes_temp = [box for box in unit if number in values[box]]
            if len(boxes_temp) == 1:
                values[boxes_temp[0]] = number
    return values


def reduce_puzzle(values):

    # Iterate through all sudoku solving strategies until the sudoku is solved
    # Args:
    #     values(dict): The sudoku in dictionary form
    # Returns:
    #     values(dict): A reduced version of the sudoku (Solved if no search required)

    stalled = False
    count = 0
    
    while not stalled:
        # Solved boxes before constraint propagation
        single_digit_boxes = [box for box in values if len(values[box]) == 1]

        values = eliminate(values)
        values = only_choice(values)

        # Solved boxes after constrained propagation
        new_signle_digit_boxes = [box for box in values if len(values[box]) == 1]

        # If no new boxes were solved after constraint propagation, then the sudoku is stalled and 
        # search needs to be implemented
        stalled = len(single_digit_boxes) == len(new_signle_digit_boxes)

        # Check for empty boxes
        if len([box for box in values if len(values[box]) == 0]):
            return False
    return values

def solved_sudoku(values):
    # Args:
    #     values(dict): The sudoku in dictionary form
    # Returns:
    #     True if the sudoku is solved, false if not

    if all(len(values[box]) == 1 for box in values):
        solved = True
        for unit in unit_list:
            for number in cols:
                boxes_temp = [box for box in unit if number in values[box]]
                if len(boxes_temp) > 1 or len(boxes_temp) == 0:
                    solved = False
        if solved:
            return solved
        else:
            return False
    else:
        return False

def search(values):
    
    # If the sudoku cannot be solved with constraint propagation alone we implement a search strategy.
    # In this strategy we find the boxes with the least available values and assign it to them, thus
    # creating a search tree with a depth-first search strategy.
    # Args:
    #     values(dict): The sudoku in dictionary form
    # Returns:
    #     values(dict): The solved sudoku or False if we didn't reach a solution
    values = reduce_puzzle(values)
    
    if values == False:
        # If values is False then there are empty boxes in the dictionary
        return False

    if solved_sudoku(values):
        # Returns the dictionary if sudoku is solved
        return values

    if not all(len(values[box]) == 1 for box in values):
        numbers_to_try, unsolved_box = min((len(values[box]), box) for box in values if len(values[box]) > 1)

        if numbers_to_try == 2:
            # Check for naked twins
            for peer in peers[unsolved_box]:
                if values[peer] == values[unsolved_box]:
                    # If naked twins are found then elimination is implemented to their common peers
                    values = naked_twins(values)
                    break

        # Try all numbers for the boxes and apply recursion
        for number in values[unsolved_box]:
            temp = values.copy()
            temp[unsolved_box] = number
            attempt = search(temp)
            if attempt and solved_sudoku(attempt):
                return attempt
            else:
                continue

def solve(grid):
    """
    Find the solution to a Sudoku grid.
    Args:
        grid(string): a string representing a sudoku grid.
            Example: '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    Returns:
        The dictionary representation of the final sudoku grid. False if no solution exists.
    """
    values = search(grid_values(grid))
    if values:
        return values
    else:
        return False

if __name__ == '__main__':
    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    display(solve(diag_sudoku_grid))

    try:
        from visualize import visualize_assignments
        # visualize_assignments(assignments)

    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
