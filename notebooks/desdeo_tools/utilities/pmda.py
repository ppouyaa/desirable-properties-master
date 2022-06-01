import numpy as np
import math





def intersection_with_axes(number):
    """
    :param number: Number of objectives : <int>
    :return: Plan intersection with axes : <list<np.arrays>>
    """
    i = []
    for j in range(number):
        array_of_zeros = np.zeros(number)
        array_of_zeros[j] = 1
        i.append(array_of_zeros)
    return i


def get_mapping_point_of_r1(vector_reference_point):
    """
    :param vector_reference_point : vector from ideal point to reference_point : <np.array>
    :return intersection of the vector with the plane S : f1 + f2 + ... + fm = 1 : <np.array>
    """
    t = 1/np.sum(vector_reference_point)
    return t * vector_reference_point


def get_q_set(vector_mapping_point_r1, epsilon_number, intersections_vectors):
    """
    :param : vector_mapping_point_r1 : intersection of the vector with the plane S : f1 + f2 + ... + fm = 1 : <np.array>
    :param : epsilon_number : given <float>
    :param : intersections_vectors : Plan intersection with axes : <list<np.arrays>>
    :return: decomposed points of R (Q set) : <list<np.array>>
    """
    q = []
    for vector in intersections_vectors:
        p = vector_mapping_point_r1 + (epsilon_number * (vector - vector_mapping_point_r1))
        q.append(p)
    q.append(vector_mapping_point_r1)
    return q





def get_mapping_points_in_second_plane(epsilon, delta_cte, vector_r1, intersections_axes):
    """
    :param : q_set_list : decomposed points of R (Q set) : <list<np.array>>
    :param : delta_cte : given <float>
    :param : vector_r1 : intersection of the vector with the plane S : f1 + f2 + ... + fm = 1 : <np.array>
    :param : intersection of the vector with the plane S : f1 + f2 + ... + fm = 1 : <np.array>
    :return: q_set_points mapped to S'
    """
    q_prime_set = []
    for intersection in intersections_axes:
        p_prime = (delta_cte * vector_r1 + (delta_cte * epsilon * (intersection - vector_r1)))
        q_prime_set.append(p_prime)
    q_prime_set.append(delta_cte * vector_r1)
    return q_prime_set





def get_distance_between_two_vect(vect1, vect2):
    """
    :param : vect1 : vector: <np.array>
    :param : vect2 : vector : <np.array>
    :return: distance :<float>
    """
    return np.linalg.norm(vect1 - vect2)


def unit_vector(vector):
    """
    :param : vector : vector : <np.array>
    :return: unit vector : <np.array>
    """
    if np.linalg.norm(vector) == 0:
        return None
    else:
        return vector / np.linalg.norm(vector)


def angle_between(vector_1, vector_2):
    """
    :param : vector_1 : vector: <np.array>
    :param : vector_2 : vector : <np.array>
    :return: angle in rad : <float>
    """
    v1_u = unit_vector(vector_1)
    v2_u = unit_vector(vector_2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def check_point_in_solutions(calculated_q_set_points, point_to_check, big_number = 1000):
    """
    :param : calculated_q_set_points : q_set_points mapped to S' <list<np.array>>
    :param : point_to_check : point to check if it is in the solutions <np.array>
    :param : big_number : this number have to be big > than the max{fi} of your individuals <int>
    :return: True or False , depending on the point if it is in the solutions (omega) or not 
    """
    simplex_on_ideal_point = np.zeros(len(calculated_q_set_points[0]))
    vertexes = [element * big_number for element in calculated_q_set_points]
    vertexes.append(simplex_on_ideal_point)
    last_element_vertexes = vertexes[-1]
    a = point_to_check - last_element_vertexes
    t_without_transpose = []
    for vector in vertexes[:-1]:
        t_without_transpose.append(vector - last_element_vertexes)
    array_t_transpose = np.array(t_without_transpose).T
    reshaped_t_transpose = array_t_transpose.reshape(array_t_transpose.shape[0], array_t_transpose.shape[1])
    inverse_t_array = np.linalg.inv(reshaped_t_transpose)
    lmbda = np.dot(inverse_t_array, a)
    rounded_lambda = np.round(lmbda, 5)
    rounded_lambda = np.append(rounded_lambda, [1 - np.sum(rounded_lambda)])
    list_boolean = np.all(rounded_lambda >= 0)
    if np.sum(rounded_lambda) <= 1 and list_boolean:
        return True
    else:
        return False


def get_distances_angles(objectives, second_plane_points, q_set_points):
    """
    :param : objectives : given points <np.arrays>
    :param : second_plane_points : q_set_points mapped to S' <list<np.array>>
    :return: distances and angles in dictionary
    """
    result = {}
    for i, individual in enumerate(objectives):
        distances = []
        angles = []
        for j, point in enumerate(second_plane_points):
            distances.append(get_distance_between_two_vect(individual, point))
        min_value_index = distances.index(min(distances))
        if check_point_in_solutions(q_set_points[:-1], second_plane_points[min_value_index]):
            angle = 0

        else:
            angle = angle_between(second_plane_points[min_value_index], individual)
        result[str(i)] = {"distance":min(distances), "angle": angle}
    return result
        

def get_pmda(reference_point,objectives, number_of_objectives,delta, epsilon):


    intersections_axes =  intersection_with_axes(number_of_objectives)
    mapping_point_r1 = get_mapping_point_of_r1(np.asarray(reference_point))
    q_set_points = get_q_set(mapping_point_r1, epsilon, intersections_axes)

    q_set = get_q_set(mapping_point_r1, epsilon, intersections_axes)
    q_set_prime = get_mapping_points_in_second_plane(epsilon, delta, mapping_point_r1, intersections_axes)
    distances_and_angles = get_distances_angles(objectives, q_set_prime, q_set_points)
    #print(distances_and_angles)
    pi = math.pi
    sum_sum = 0
    for d_a in distances_and_angles:
        sum_sum = sum_sum + distances_and_angles[d_a]['distance'] + (distances_and_angles[d_a]['angle'] / pi)
    pmda_p = sum_sum / len(distances_and_angles)
    return pmda_p

# np.random.seed(42) # this help as to get the same random np.array for the "given" reference point 
# number_of_objectives = 5
# reference_point = np.random.rand(number_of_objectives) # reference point is a given point
# epsilon = 0.1 
# delta = 0.01 # delta is a given float
# objectives = []
# for number in range(number_of_objectives + 1):
#     objectives.append(np.random.rand(number_of_objectives))

# print(get_pmda(reference_point,objectives, number_of_objectives,delta, epsilon))