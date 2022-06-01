# from desdeo_emo.othertools.ReferenceVectors import ReferenceVectors
import numpy as np
import baseADM


def generateRP4learning(base: baseADM):

    ideal_cf = base.ideal_point

    translated_cf = base.translated_front

    # Assigment of the solutions to the vectors
    assigned_vectors = base.assigned_vectors

    # Find the vector which has a minimum number of assigned solutions
    number_assigned = np.bincount(assigned_vectors)
    min_assigned_vector = np.atleast_1d(
        np.squeeze(
            np.where(
                number_assigned == np.min(number_assigned[np.nonzero(number_assigned)])
            )
        )
    )
    sub_population_index = np.atleast_1d(
        np.squeeze(np.where(assigned_vectors == min_assigned_vector[0]))
        # If there are multiple vectors which have the minimum number of solutions, first one's index is used
    )
    # Assigned solutions to the vector which has a minimum number of solutions
    sub_population_fitness = translated_cf[sub_population_index]
    # Distances of these solutions to the origin
    sub_pop_fitness_magnitude = np.sqrt(
        np.sum(np.power(sub_population_fitness, 2), axis=1)
    )
    # Index of the solution which has a minimum distance to the origin
    minidx = np.where(sub_pop_fitness_magnitude == np.nanmin(sub_pop_fitness_magnitude))
    distance_selected = sub_pop_fitness_magnitude[minidx]

    # Create the reference point
    reference_point = distance_selected[0] * base.vectors.values[min_assigned_vector[0]]
    reference_point = np.squeeze(reference_point + ideal_cf)
    # reference_point = reference_point + ideal_cf
    return reference_point


def get_max_assigned_vector(assigned_vectors):

    number_assigned = np.bincount(assigned_vectors)
    max_assigned_vector = np.atleast_1d(
        np.squeeze(
            np.where(
                number_assigned == np.max(number_assigned[np.nonzero(number_assigned)])
            )
        )
    )
    return max_assigned_vector


def generateRP4decision(base: baseADM, max_assigned_vector):

    assigned_vectors = base.assigned_vectors

    ideal_cf = base.ideal_point

    translated_cf = base.translated_front

    sub_population_index = np.atleast_1d(
        np.squeeze(np.where(assigned_vectors == max_assigned_vector))
    )
    sub_population_fitness = translated_cf[sub_population_index]
    # Distances of these solutions to the origin
    sub_pop_fitness_magnitude = np.sqrt(
        np.sum(np.power(sub_population_fitness, 2), axis=1)
    )
    # Index of the solution which has a minimum distance to the origin
    minidx = np.where(sub_pop_fitness_magnitude == np.nanmin(sub_pop_fitness_magnitude))
    distance_selected = sub_pop_fitness_magnitude[minidx]

    # Create the reference point
    reference_point = distance_selected[0] * base.vectors.values[max_assigned_vector]
    reference_point = np.squeeze(reference_point + ideal_cf)
    # reference_point = reference_point + ideal_cf
    return reference_point

def generatePerturbatedRP4decision(base: baseADM, max_assigned_vector):

    assigned_vectors = base.assigned_vectors
    #theta = base.theta

    ideal_cf = base.ideal_point

    translated_cf = base.translated_front

    sub_population_index = np.atleast_1d(
        np.squeeze(np.where(assigned_vectors == max_assigned_vector))
    )
    sub_population_fitness = translated_cf[sub_population_index]

    # angles = theta[sub_population_index, max_assigned_vector]
    # angles = np.divide(angles)
    # print(angles)
    # Distances of these solutions to the origin
    sub_pop_fitness_magnitude = np.sqrt(
        np.sum(np.power(sub_population_fitness, 2), axis=1)
    )
    # Index of the solution which has a minimum distance to the origin
    minidx = np.where(sub_pop_fitness_magnitude == np.nanmin(sub_pop_fitness_magnitude))
    distance_selected = sub_pop_fitness_magnitude[minidx]

    # aminidx = np.where(angles == np.nanmin(angles))

    # Create the reference point
    reference_point = distance_selected[0] * base.vectors.values[max_assigned_vector]

    # Find the distance from the nearest solution to the reference point
    distance = min(np.linalg.norm(reference_point - i) for i in sub_population_fitness)

    # nearest = np.squeeze(sub_population_fitness[aminidx] + ideal_cf)

    # distance = np.linalg.norm(nearest - reference_point)
    # print("distance", distance)

    reference_point = np.squeeze(reference_point + ideal_cf)

    reference_point = np.squeeze(reference_point - distance)

    # The following line is to make sure that the components of the reference point cannot be smaller than the components of the ideal point
    # update the following line if the ideal point is not zero
    reference_point[reference_point < 0] = np.finfo(float).eps
    # print(reference_point)

    return reference_point