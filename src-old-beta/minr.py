##################################################################################
## Proof-of-concept ## Proof-of-concept ## Proof-of-concept ## Proof-of-concept ## 
##################################################################################
########################### Coded by andres.moran.l@uc.cl ########################
##################################################################################

import _pickle as cPickle 
import os

from sage.all import * 

import functools 

from config import MAX_NUMBER_OF_ROWS, MAX_NUMBER_OF_COLS

def DBG(*var):
    print('-------------------------')
    print(f"{var=}")
    print('-------------------------')

def printl(list):
    for item in list: print(item)

def factorial(n): 
    r = 1
    if n > 1:
        for i in range(2, n + 1):
            r = r*i
    else:
        r = 1
    return r

def bin_coeff(n, k): 
    return factorial(n) // (factorial(k)*factorial(n-k))

def is_admissible(list_monomial_power): 
    r = True

    len_list_monomial_power = len(list_monomial_power)

    for i in range(0, len_list_monomial_power):
        if i + 1 < len_list_monomial_power:
            r = r and (list_monomial_power[i] >= 2*list_monomial_power[i + 1])
        else:
            break ## TODO: rewrite

    return r

def search_non_admissible(list_monomial_power):
    r = -1 
    
    len_list_monomial_power = len(list_monomial_power)

    for i in range(0, len_list_monomial_power):
        if i + 1 < len_list_monomial_power:
            if list_monomial_power[i] < 2*list_monomial_power[i + 1]:
                r = i
                break ## TODO: rewrite

    return r

def cancel_mod_2(list_list_linear_combination):
    list_clean_linear_combination = []

    for i in range(0, len(list_list_linear_combination)):
        repeated_occurences = 0
        for j in range(0, len(list_list_linear_combination)):
                if list_list_linear_combination[i] == list_list_linear_combination[j]:
                    repeated_occurences += 1

        if repeated_occurences % 2 == 1:
            if not list_list_linear_combination[i] in list_clean_linear_combination: 
                list_clean_linear_combination.append(list_list_linear_combination[i])

    return list_clean_linear_combination

def adem_relation(list_monomial_power): 
    list_list_monomial_power = [] 

    if is_admissible(list_monomial_power):
        list_list_monomial_power.append(list_monomial_power)
    else:
        a = list_monomial_power[0]
        b = list_monomial_power[1]
        for i in range(max(a - b + 1, 0), (a >> 1) + 1): 
            c = bin_coeff(b - i - 1, a - 2*i)
            if not (c & 0X00000001): 
                continue
            else:
                if i == 0:
                    list_list_monomial_power.append([a + b - i])
                else:
                    list_list_monomial_power.append([a + b - i, i])

    return cancel_mod_2(list_list_monomial_power) 

def right_distribute_sq_product(list_left_chunk, list_list_linear_combination): 
    list_list_linear_combination_steenrod_powers = []

    for list_monomial in list_list_linear_combination:
        list_list_linear_combination_steenrod_powers.append(list_left_chunk + list_monomial)

    return cancel_mod_2(list_list_linear_combination_steenrod_powers)

def left_distribute_sq_product(list_list_linear_combination, list_right_chunk): 
    list_list_linear_combination_steenrod_powers = []

    for list_monomial in list_list_linear_combination:
        list_list_linear_combination_steenrod_powers.append(list_monomial + list_right_chunk)

    return cancel_mod_2(list_list_linear_combination_steenrod_powers)

def write_as_admissible_linear_combination(list_monomial_power):
    list_list_monomial_power = [] 
    
    if is_admissible(list_monomial_power):
       list_list_monomial_power.append(list_monomial_power)
    else:
        len_list_monomial_power = len(list_monomial_power)
        if len_list_monomial_power > 2:
            non_admissible_location = search_non_admissible(list_monomial_power)
            
            admissible_linear_combination = adem_relation([
                    list_monomial_power[non_admissible_location], 
                    list_monomial_power[non_admissible_location + 1]
            ])

            list_monomial_left_chunk = list_monomial_power[:non_admissible_location]
            list_monomial_right_chunk = list_monomial_power[non_admissible_location+2:]
            list_left_factor = right_distribute_sq_product(list_monomial_left_chunk, admissible_linear_combination)
            list_expanded_product = left_distribute_sq_product(list_left_factor, list_monomial_right_chunk)
        
            for list_monomial in list_expanded_product:
                list_list_monomial_power = list_list_monomial_power + write_as_admissible_linear_combination(list_monomial)

        elif len_list_monomial_power == 2:
            list_list_monomial_power = adem_relation(list_monomial_power)
        
        else:
            pass 

    return cancel_mod_2(list_list_monomial_power)

def bin_length(k):
    r = k
    k_binary_length = 0 
    while r > 0:
        r = r >> 1
        k_binary_length += 1

    return k_binary_length

def get_admissible_generators_for_degree(k, k_binary_length = -1): 
    list_list_generators = []

    if k_binary_length == -1: 
        k_binary_length = bin_length(k) 

    for fixed_length_product_monomials in range(1, k_binary_length + 1):
        if fixed_length_product_monomials == 1:
            list_list_generators.append([k])
        else:
            for biggest_power in range(2**(fixed_length_product_monomials-1), k - fixed_length_product_monomials + 2):
                list_list_admissible_generators = get_admissible_generators_for_degree(k-biggest_power, fixed_length_product_monomials - 1)
                            
                for j in range(0, len(list_list_admissible_generators)):
                    list_list_admissible_generators[j].insert(0, biggest_power)

                    if is_admissible(list_list_admissible_generators[j]):
                        if not list_list_admissible_generators[j] in list_list_generators:
                            list_list_generators.append(list_list_admissible_generators[j])
                        
    return list_list_generators

def get_generator_zero_row(column):
    return [column] 

def multiply_by_right_at_deg(deg, sq_concat):
    list_generators = []

    if deg == 0:
        list_generators = [[sq_concat]]
    else:
        dim_deg = len(list_admissible_generators_by_deg[deg]) 

        for k in range(0, dim_deg):
            list_generators.append(list_admissible_generators_by_deg[deg][k].copy()) 
            list_generators[k].append(sq_concat)
        
    return list_generators

def get_admissible_generators_for_coord(x, y):
    list_generators = []

    if x == 0:
        list_generators = [k for k in list_admissible_generators_by_deg[y]] 
    else:
        list_generators = get_admissible_generators_in_column(x, y)

    return list_generators

def get_admissible_generators_in_column(col_number, deg):
    list_generators = []

    list_generators = multiply_by_right_at_deg(deg, (col_number, 1))

    number_of_new_generators = sum([
        1 if list_found_generators[col_number][k][0] - col_number <= deg else 0 for k in range(0, len(list_found_generators[col_number]))
    ])

    for i in range(1, number_of_new_generators): 
        list_generators = list_generators \
            + multiply_by_right_at_deg(deg - (list_found_generators[col_number][i][0] - col_number), list_found_generators[col_number][i])

    return list_generators

def construct_matrix_differential(list_dom_generators, list_dom_generators_eval, list_cod_generators):
    len_dom_generators = len(list_dom_generators)
    len_cod_generators = len(list_cod_generators)

    list_list_differential = [[0]*len_dom_generators for k in range(0, len_cod_generators)]
    
    for i in range(0, len_dom_generators):
        if len(list_dom_generators_eval[i]) == 0:
            continue

        for j in range(0, len(list_dom_generators_eval[i])):
            for k_cod_basis_element in range(0, len_cod_generators):

                list_monomial_img_base_element = list_dom_generators_eval[i][j]

                if list_monomial_img_base_element == list_cod_generators[k_cod_basis_element][0]:
                    list_list_differential[k_cod_basis_element][i] = 1

    return list_list_differential

def construct_matrix_differential_gen(list_dom_generators, list_dom_generators_eval, list_cod_generators):
    len_dom_generators = len(list_dom_generators)
    len_cod_generators = len(list_cod_generators)

    list_list_differential = [[0]*len_dom_generators for k in range(0, len_cod_generators)]

    for i in range(0, len_dom_generators):
        if len(list_dom_generators_eval[i]) == 0:
            continue

        for j in range(0, len(list_dom_generators_eval[i])):
            for k in range(0, len(list_dom_generators_eval[i][j])):
                list_monomial_img_base_element = list_dom_generators_eval[i][j][k]

                for k_cod_basis_element in range(0, len_cod_generators):
                    
                    if list_monomial_img_base_element == list_cod_generators[k_cod_basis_element]:
                        list_list_differential[k_cod_basis_element][i] = 1

    return list_list_differential

def compute_kernel(x_dom, y_dom, x_cod, y_cod):
    list_kernel = []

    list_dom_generators_eval = []
    list_dom_generators = get_admissible_generators_for_coord(x_dom, y_dom)
    list_cod_generators = get_admissible_generators_for_coord(x_cod, y_cod)
    
    if x_dom == 1: 
        for k in range(0, len(list_dom_generators)):
            list_dom_generators_eval.append(write_as_admissible_linear_combination(list_dom_generators[k][:-1] + [list_dom_generators[k][-1][0]])) 
            
        for k in range(0, len(list_cod_generators)):
            list_cod_generators[k] = write_as_admissible_linear_combination(list_cod_generators[k])

        matrix_differential = construct_matrix_differential(list_dom_generators, list_dom_generators_eval, list_cod_generators)
    else:
        list_dom_generators_eval = eval_diff(x_dom, list_dom_generators)
        matrix_differential = construct_matrix_differential_gen(list_dom_generators, list_dom_generators_eval, list_cod_generators)
    
    sage_matrix = Matrix(GF(2), matrix_differential, sparse=True)
    list_kernel = list(sage_matrix.right_kernel().basis())
    
    return {
        'domainCoord': (x_dom, y_dom),  
        'restrDifferential': matrix_differential,
        'domainBasis': list_dom_generators,
        'codomainBasis': list_cod_generators,
        'kernelBasis': list_kernel,
    }

def eval_diff(col_number, list_basis): 
    list_images = []
    
    for basis_monomial in list_basis:
        basis_tracker = basis_monomial[-1]
        
        bool_item_found = False 
        k = 0
        while not bool_item_found:
            if list_new_generator_mapping_table[col_number][k][0] == basis_tracker:
                bool_item_found = True
            else:
                k += 1
        
        list_linear_comb_for_basis_element = []
        for dest_monomial_associated in list_new_generator_mapping_table[col_number][k][1]: 
            expression_to_reduce = basis_monomial[:-1] + dest_monomial_associated[:-1] 

            for list_monomial in write_as_admissible_linear_combination(expression_to_reduce):
                if not [list_monomial + [dest_monomial_associated[-1]]] in list_linear_comb_for_basis_element:
                    list_linear_comb_for_basis_element.append([list_monomial + [dest_monomial_associated[-1]]])
                else:
                    list_linear_comb_for_basis_element.remove([list_monomial + [dest_monomial_associated[-1]]])

        list_images.append(list_linear_comb_for_basis_element)
    
    return list_images

def survivors_to_linear_comb(matrix, list_basis):
    list_linear_comb = []

    for row in matrix:
        for i in range(0, len(row)):
            if row[i] > 0:
                list_linear_comb.append(list_basis[i])

    return list_linear_comb

def compute_free_resolution():
    col_start = 2
    deg_start = 1

    for i in range(col_start, MAX_NUMBER_OF_COLS): 
        list_found_generators.append([(i, 1)]) 
        list_new_generator_mapping_table.append( 
            [(  (i, 1), [[1, (i-1, 1)]]  )] 
        )
        count_new_generators = 1

        for j in range(deg_start, MAX_NUMBER_OF_ROWS - i + 1): 
            dict_kernel_computation = compute_kernel(i - 1, j + 1, i - 2, j + 2) 
            list_differentials.append(dict_kernel_computation)
            
            if len(dict_kernel_computation['kernelBasis']) >= 0:
                current_generators = get_admissible_generators_for_coord(i, j)
                
                kernel_generators = dict_kernel_computation['kernelBasis'].copy()
                domain_generators = dict_kernel_computation['domainBasis'].copy()
                
                kernel_generators_survivors = []

                if len(kernel_generators) == 0:
                    continue 
                
                list_monomial_img_as_vect = [[0]*len(kernel_generators[0]) for k in range(0, len(current_generators))]
                
                eval_diff_curr_gen = eval_diff(i, current_generators)
                len_eval_diff_curr_gen = len(eval_diff_curr_gen)
                
                for u in range(0, len_eval_diff_curr_gen):
                    list_basis_element_img = eval_diff_curr_gen[u]
                    
                    for list_linear_combination_img in list_basis_element_img:
                        for monomial_img in list_linear_combination_img:
                                
                            for s in range(0, len(domain_generators)):
                                if domain_generators[s] == monomial_img:
                                    list_monomial_img_as_vect[u][s] = 1
                
                span_monomial_img = span(list_monomial_img_as_vect, GF(2))
                
                quotient_kernel = span(kernel_generators, GF(2)).quotient(span_monomial_img)
                for kernel_survivor in quotient_kernel:
                    if not sum(list(kernel_survivor)) == 0: 
                        kernel_generators_survivors.append(quotient_kernel.lift(kernel_survivor))
                
                kernel_dim_to_kill = len(kernel_generators_survivors)
                if kernel_dim_to_kill > 0:
                    deg_new_generator = i + j
                    
                    for r in range(1, kernel_dim_to_kill + 1):
                        list_new_generator_mapping_table[i].append(
                            ((deg_new_generator, r), survivors_to_linear_comb([kernel_generators_survivors[r-1]], domain_generators))
                        )
                        list_found_generators[i].append((deg_new_generator, r))
                    
                    DBG('[*] New generator(s): ', i, j, 'dimension: ' + str(kernel_dim_to_kill), 'degree:', str(i + j))
                    
    return 

list_admissible_generators_by_deg = []
list_found_generators = [(0, 1), [(2**i, 1) for i in range(0, bin_length(MAX_NUMBER_OF_ROWS + 1) )]]
list_new_generator_mapping_table = [[( )],[( )],] 
list_differentials = []

if not os.path.exists('list_admissible_generators_by_deg.list'): 
    list_admissible_generators_by_deg = [ [[0]] ]  
    for k in range(1, MAX_NUMBER_OF_ROWS + 1):
        list_deg_k_generators = get_admissible_generators_for_degree(k) 
        list_admissible_generators_by_deg.append(list_deg_k_generators)

        print('Saving computations in degree ' + str(k))

    with open('list_admissible_generators_by_deg.list', 'wb') as file_list_admissible_generators_by_deg:
        cPickle.dump(list_admissible_generators_by_deg, file_list_admissible_generators_by_deg)
else:
    with open('list_admissible_generators_by_deg.list', 'rb') as file_list_admissible_generators_by_deg:
        list_admissible_generators_by_deg = cPickle.load(file_list_admissible_generators_by_deg)

compute_free_resolution() 

if not os.path.exists('dict_truncated_resolution.list'): 
    dict_truncated_resolution = {
        'list_admissible_generators_by_deg': list_admissible_generators_by_deg,
        'list_found_generators': list_found_generators,
        'list_new_generator_mapping_table': list_new_generator_mapping_table,
        'list_differentials': list_differentials
    }
    with open('dict_truncated_resolution.list', 'wb') as file_dict_truncated_resolution:
        cPickle.dump(dict_truncated_resolution, file_dict_truncated_resolution)

    print('Saved computations.')

else:
    with open('dict_truncated_resolution.list', 'rb') as file_dict_truncated_resolution:
        dict_truncated_resolution = cPickle.load(file_dict_truncated_resolution)

    print('Loaded computations.')