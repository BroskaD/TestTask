from typing import Optional

import numpy as np


def read_input_matrix() -> Optional[np.ndarray]:
    """
    Reads and parses users input to matrix
    :return: Result matrix
    """

    try:
        matrix_size = input('Enter M and N (whitespace as separator): ').strip()
        matrix_size = [int(x) for x in matrix_size.split(' ')]

        raw_matrix = list()

        print('Enter matrix (whitespace as separator, enter as new row)')
        for _ in range(matrix_size[0]):
            raw_matrix.append([int(x) for x in input().strip().split(' ')])

        matrix = np.array(raw_matrix, dtype=np.uint8)

    except:
        return

    return matrix


def create_index_shift_matrix(window_size: int = 3) -> np.ndarray:
    """
    Creates indexes shift to return from window indexes to original matrix indexes
    :param window_size: Size of window
    :return: Array with shape of [window_size, window_size, 2] where [x, y] element represents index shift according to
    the center of matrix
    """

    indexes_range = np.linspace(-(window_size // 2), window_size // 2, window_size, dtype=int)
    xx, yy = np.meshgrid(indexes_range, indexes_range)
    index_mask = np.stack((yy, xx), axis=-1)
    return index_mask


def cut_window(matrix: np.ndarray, location: tuple, window_size: int = 3) -> np.ndarray:
    """
    Cuts original matrix to [window_size, window_size] with center in location
    :param matrix: Original matrix
    :param location: Center of the sliced window
    :param window_size: Window size
    :return: Matrix with [window_size, window_size] shape
    """

    result = matrix[(location[0] - window_size // 2):(location[0] + window_size // 2) + 1,
                    (location[1] - window_size // 2):(location[1] + window_size // 2) + 1]
    return result


def count_islands(matrix: np.ndarray) -> int:
    """
    Counts amount of islands according to the task logic
    :param matrix: Input matrix (map)
    :return: Amount of separate islands
    """

    # Pad input matrix to process edge islands correctly
    padded_matrix = np.pad(matrix, pad_width=1, mode='constant', constant_values=0)
    non_zero_indexes = np.transpose(np.nonzero(padded_matrix))

    visited_indexes = list()
    all_non_zero_indexes = [tuple(x) for x in non_zero_indexes.tolist()]
    island_counter = 0

    index_shift_matrix = create_index_shift_matrix()
    # Kernel for island connectivity logic
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)

    for location in all_non_zero_indexes:

        if location in visited_indexes:
            continue

        local_area = cut_window(padded_matrix, location)
        masked_area = local_area * kernel

        # Single island
        if np.sum(masked_area).item() == 0:
            island_counter += 1
            visited_indexes.append(location)
            continue

        # Process advanced island
        new_next_locations = [location]
        while new_next_locations:
            current_position = new_next_locations.pop(0)
            local_area = cut_window(padded_matrix, current_position)
            masked_area = local_area * kernel
            visited_indexes.append(current_position)
            # New potential area indexes are just indexes where product of window and kernel isn't zero
            new_potential_area_indexes = [x for x in np.transpose(np.nonzero(masked_area))]
            # Return to the input matrix indexes
            new_potential_indexes = [np.array(current_position) + index_shift_matrix[ind[0], ind[1]]
                                     for ind in new_potential_area_indexes]
            new_potential_indexes = [tuple(x.tolist()) for x in new_potential_indexes]

            for new_index in new_potential_indexes:
                if new_index not in visited_indexes:
                    new_next_locations.append(new_index)

        island_counter += 1

    return island_counter


def main():
    matrix = read_input_matrix()

    if matrix is None:
        print('Wrong input')
        return

    result = count_islands(matrix)
    print(f'Total amount of islands: {result}')


if __name__ == '__main__':
    main()


