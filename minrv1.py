# Copyright (C) 2024 Andrés Morán - All Rights Reserved.
# %License%
# (proof-of-concept)

## TODO: mejorar notación, etc

import _pickle as cPickle
import os
import sys  # ~unused, sys.maxsize

import time
import gc  # ~unused

#######from sage.all import * # WARNING (BAD PERFORMANCE)
from sage.rings.finite_rings.finite_field_constructor import (
    GF,
)  # WARNING (BAD PERFORMANCE)
from sage.matrix.constructor import matrix as Matrix  # WARNING (BAD PERFORMANCE)
from sage.algebras.steenrod.steenrod_algebra import (
    SteenrodAlgebra,
)  # WARNING (BAD PERFORMANCE)

import plotly.graph_objects as go

import jsons

import multiprocessing

from functools import cache

# Global parameters

BOOL_COMPUTE_ONLY_ADDITIVE_STRUCTURE = False

FIXED_PRIME_NUMBER = 3

MAX_NUMBER_OF_RELATIVE_DEGREES = 130  ## 130
MAX_NUMBER_OF_MODULES = MAX_NUMBER_OF_RELATIVE_DEGREES

NUMBER_OF_THREADS = 200  # 100
DEFAULT_YONEDA_PRODUCT_MAX_DEG = 20  # DEPRECATED

UI_SHIFT_MULTIPLE_GENERATORS = 0.1

## UTILS


def DBG(*var):
    print("-" * 120)
    print(f"{var=}")
    print("-" * 120)


def printl(list):
    for item in list:
        print(item)


def print_banner():
    print_header("[[ MinimalResolution v1 - andres.moran.l@uc.cl ]]", "#", True)


def print_header(str_txt, char_delim, bool_print_lateral):
    number_of_characters = 120

    padding_length = (number_of_characters - len(str_txt)) // 2 - 1
    padding_tweak = (number_of_characters - len(str_txt)) % 2

    print(char_delim * number_of_characters)

    char_delim_lateral = char_delim
    if not bool_print_lateral:
        char_delim_lateral = " "

    print(
        char_delim_lateral
        + " " * padding_length
        + str_txt
        + " " * (padding_length + padding_tweak)
        + char_delim_lateral
    )
    print(char_delim * number_of_characters)


def log(str_event):
    str_log_file = "log.txt"
    with open(str_log_file, "a") as file_log:
        file_log.write(f"{str_event}\n")


def dump_object(obj, str_name):
    str_save_path = f"minimal_resolution_{str_name}.obj"
    if not os.path.exists(str_save_path):
        with open(str_save_path, "wb") as file_minimalresolution:
            cPickle.dump(obj, file_minimalresolution)

    print("#" * 120)
    print(f"[*] Minimal resolution object dumped to ./{str_save_path}.")
    print("#" * 120)


def load_object(str_name):
    str_loaded_path = f"minimal_resolution_{str_name}.obj"
    if os.path.exists(str_loaded_path):
        with open(str_loaded_path, "rb") as file_minimalresolution:
            r = cPickle.load(file_minimalresolution)

            print("#" * 120)
            print(f"[*] Minimal resolution object loaded from ./{str_loaded_path}.")
            print("#" * 120)
    else:
        r = False
    return r


@cache
def factorial(n):
    r = 1
    if n > 1:
        for i in range(2, n + 1):
            r = r * i
    else:
        r = 1
    return r


@cache
def bin_coeff(n, k):
    if (n, k) == (0, 0):
        return 1
    if k > n:
        return 0
    if k < 0:
        return 0
    return factorial(n) // (factorial(k) * factorial(n - k))


## CLASSES


class FPModule:
    def __init__(self, str_name, callback_generators, callback_relations, max_deg):
        self.str_name = str_name
        self.callback_generators = lambda x, y: callback_generators(x, y, max_deg)
        self.callback_relations = lambda x, y: callback_relations(x, y, max_deg)


class FPMap:
    def __init__(
        self,
        list_domain_generators,
        list_list_images,
        list_tuple_domain,
        list_tuple_codomain,
    ):  # ordered
        self.list_domain_generators = list_domain_generators
        self.list_list_images = list_list_images

        self.list_tuple_domain = list_tuple_domain
        self.list_tuple_codomain = list_tuple_codomain

    def eval(self, linear_comb):
        output_linear_comb = []

        for summand in linear_comb:
            for i in range(0, len(self.list_domain_generators)):
                if summand.generator == self.list_domain_generators[i].generator:
                    for element_img in self.list_list_images[i]:
                        output_linear_comb.append(
                            Element(
                                summand.cohomology_operation
                                * element_img.cohomology_operation,
                                element_img.generator,
                            )
                        )

        return output_linear_comb

    def __repr__(self):
        return f"{self.list_domain_generators} -> {self.list_list_images}"

    def __str__(self):
        return f"{self.list_domain_generators} -> {self.list_list_images}"


class YonedaProduct:
    def __init__(
        self, external_generator, internal_generator, linear_comb_output_generators
    ):
        self.external_generator = external_generator
        self.internal_generator = internal_generator
        self.linear_comb_output_generators = linear_comb_output_generators

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"({self.external_generator})*({self.internal_generator}) == ({self.linear_comb_output_generators})"


class GradedMorphism:
    def __init__(self, list_morphisms):
        self.list_morphisms = list_morphisms

    def get_morphism_from_bigraded_domain(self, tuple_dom):
        for morphism in self.list_morphisms:
            if morphism.tuple_dom == tuple_dom:
                return morphism
        return -1

    def eval_linear_comb(self, list_list_linear_comb, tuple_dom):
        morphism = self.get_morphism_from_bigraded_domain(tuple_dom)
        if morphism == -1:
            return -1

        return [morphism.eval_linear_comb(list_list_linear_comb), morphism.tuple_cod]

    def eval_element(self, element, tuple_dom):
        morphism = self.get_morphism_from_bigraded_domain(tuple_dom)
        if morphism == -1:
            return -1

        return [morphism.eval_element(element), morphism.tuple_cod]

    def eval_vector(self, list_vector, tuple_dom):
        morphism = self.get_morphism_from_bigraded_domain(tuple_dom)
        if morphism == -1:
            return [[], -1]

        return [morphism.eval_vector(list_vector), morphism.tuple_cod]

    def __repr__(self):
        s = ""
        for morphism in self.list_morphisms:
            s += str(morphism) + "\n"
        return s

    def __str__(self):
        s = ""
        for morphism in self.list_morphisms:
            s += str(morphism) + "\n"
        return s


class AlgebraGeneratorRelation:
    def __init__(self, module_element, steenrod_operation, list_sum_output):
        self.module_element = module_element
        self.steenrod_operation = steenrod_operation
        self.list_sum_output = list_sum_output

    def __eq__(self, other):
        return (
            self.module_element == other.module_element
            and self.steenrod_operation == other.steenrod_operation
            and self.list_sum_output == other.list_sum_output
        )

    def __repr__(self):
        return f"{self.steenrod_operation} (({self.module_element})) = {self.list_sum_output}"

    def __str__(self):
        return f"{self.steenrod_operation} (({self.module_element})) = {self.list_sum_output}"


class ExtendedAlgebraGenerator:  # TODO: merge
    def __init__(self, module_index, deg, index, bool_is_free, str_alias):
        self.module_index = module_index
        self.deg = deg
        self.index = index
        self.bool_is_free = bool_is_free
        self.str_alias = str_alias

    def __eq__(self, other):
        return (
            self.module_index == other.module_index
            and self.deg == other.deg
            and self.index == other.index
        )

    def __repr__(self):
        return f"{self.module_index}, {self.deg}, {self.index};; is_free: {self.bool_is_free}\t  @ str_alias: {self.str_alias}"

    def __str__(self):
        return f"{self.module_index}, {self.deg}, {self.index};; is_free: {self.bool_is_free}\t@ str_alias: {self.str_alias}"


class AlgebraGenerator:  # TODO: merge
    def __init__(self, module_index, deg, index):
        self.module_index = module_index
        self.deg = deg
        self.index = index

    def __eq__(self, other):
        return (
            self.module_index == other.module_index
            and self.deg == other.deg
            and self.index == other.index
        )

    def __repr__(self):
        return f"{self.module_index}, {self.deg}, {self.index}"

    def __str__(self):
        return f"{self.module_index}, {self.deg}, {self.index}"

    def __hash__(self):
        return hash(str(self))


class Element:
    def __init__(self, cohomology_operation, generator):
        self.cohomology_operation = cohomology_operation
        self.generator = generator

    def deg(self):
        if self.cohomology_operation == 0:
            return -1
        return self.cohomology_operation.degree() + self.generator.deg

    def __eq__(self, other):
        if (
            not self.cohomology_operation.is_zero()
            and not other.cohomology_operation.is_zero()
        ):
            if (
                self.cohomology_operation.degree() == 0
                and other.cohomology_operation.degree() == 0
            ):
                if self.generator == other.generator:
                    if (
                        self.cohomology_operation.leading_coefficient()
                        == other.cohomology_operation.leading_coefficient()
                    ):
                        return True
        return (
            self.cohomology_operation == other.cohomology_operation
            and self.generator == other.generator
        )

    def __str__(self):
        return f"{self.cohomology_operation}; {self.generator}"

    def __repr__(self):
        return f"{self.cohomology_operation}; {self.generator}"

    def __add__(self, other):
        if self.generator == other.generator:
            return Element(
                self.cohomology_operation + other.cohomology_operation, self.generator
            )
        else:
            return [self.cohomology_operation, other.cohomology_operation]

    def __hash__(self):
        return hash(str(self))

    def encode(self):
        return self.__dict__


class MTEntry:
    def __init__(self, src, list_dst):
        self.src = src
        self.list_dst = list_dst

    def __repr__(self):
        return f"src: {self.src}, dst: {self.list_dst}"


class Morphism:
    def __init__(
        self,
        fixed_prime,
        list_dom_basis,
        list_cod_basis,
        list_list_images,
        tuple_dom=(-1, -1),
        tuple_cod=(-1, -1),
    ):  ## TODO: diff. coord.
        self.list_dom_basis = self.sanitizeRedundant(list_dom_basis)
        self.list_cod_basis = self.sanitizeRedundant(list_cod_basis)
        self.list_list_images = list_list_images

        self.fixed_prime = fixed_prime

        self.matrix = Matrix(
            GF(fixed_prime),
            self.getListListMatrix(list_dom_basis, list_cod_basis, list_list_images),
            sparse=True,
        )

        self.tuple_dom = tuple_dom
        self.tuple_cod = tuple_cod

    def eval_element(self, element):
        return self.eval_vector(self.domainLinearCombToVector([[element]]))

    def eval_vector(self, list_vector):
        return self.matrix * Matrix(GF(self.fixed_prime), list_vector, sparse=True)

    def eval_linear_comb(self, list_list_linear_comb):
        element_as_vector = self.convertElementToVector(
            [list_list_linear_comb], self.list_dom_basis
        )
        return self.matrix * Matrix(
            GF(self.fixed_prime), element_as_vector, sparse=True
        )

    def convertLinearCombToVector(self, list_list_linear_comb, list_basis):
        return self.getListListMatrix([0], list_basis, list_list_linear_comb)[0]

    def domainLinearCombToVector(self, list_list_linear_comb):
        return self.getListListMatrix([0], self.list_dom_basis, list_list_linear_comb)

    def codomainLinearCombToVector(self, list_list_linear_comb):
        return self.getListListMatrix([0], self.list_cod_basis, list_list_linear_comb)[
            0
        ]

    def __repr__(self):
        str_matrix = str(self.matrix)
        return f"{[self.tuple_dom, self.tuple_cod]} MATRIX: \n{str_matrix}\n"

    def sanitizeRedundant(self, list_basis):
        return [
            element for element in list_basis if not element.cohomology_operation == 0
        ]

    def OLDgetListListMatrix(
        self, list_dom_basis, list_cod_basis, list_list_images
    ):  ## TODO: remove
        dim_cod = len(list_cod_basis)
        dim_dom = len(list_dom_basis)  # = len(list_list_images)

        list_list_matrix = [[0] * max(1, dim_dom) for k in range(0, max(1, dim_cod))]

        for i in range(0, dim_dom):
            for j in range(0, dim_cod):
                for k in range(0, len(list_list_images[i])):
                    for monomial_index in range(
                        0, len(list_list_images[i][k].cohomology_operation.terms())
                    ):
                        monomial = list_list_images[i][k].cohomology_operation.terms()[
                            monomial_index
                        ]
                        monomial_coefficient = monomial.leading_coefficient()

                        if (
                            list_list_images[i][k].generator
                            == list_cod_basis[j].generator
                        ):
                            bool_monomial_patch = False  ## TODO: warning...
                            if (
                                monomial.leading_monomial().trailing_support()
                                in [tuple([]), (0,)]
                                and list_cod_basis[j]
                                .cohomology_operation.leading_monomial()
                                .trailing_support()
                                in [tuple([]), (0,)]
                            ):
                                bool_monomial_patch = True
                            if (
                                monomial.leading_monomial()
                                == list_cod_basis[
                                    j
                                ].cohomology_operation.leading_monomial()
                                or bool_monomial_patch
                            ):
                                list_list_matrix[j][i] += (
                                    monomial_coefficient
                                    / list_cod_basis[
                                        j
                                    ].cohomology_operation.leading_coefficient()
                                )  ############ 1, ..., p-1

        return list_list_matrix

    def getListListMatrix(self, list_dom_basis, list_cod_basis, list_list_images):
        dim_cod = len(list_cod_basis)
        dim_dom = len(list_dom_basis)  # = len(list_list_images)

        list_list_matrix = [[0] * max(1, dim_dom) for k in range(0, max(1, dim_cod))]

        for i in range(0, dim_dom):
            for j in range(0, dim_cod):
                cod_basis_trailing_support = list_cod_basis[
                    j
                ].cohomology_operation.trailing_support()
                cod_basis_leading_coeff = list_cod_basis[
                    j
                ].cohomology_operation.leading_coefficient()

                for k in range(0, len(list_list_images[i])):
                    image_cohomology_operation = list_list_images[i][
                        k
                    ].cohomology_operation
                    dict_image_cohomology_operation_support = (
                        image_cohomology_operation.support()._mapping
                    )
                    list_keys_dict_image_cohomology_operation_support = list(
                        dict_image_cohomology_operation_support.keys()
                    )

                    for monomial_index in range(
                        0, len(list_keys_dict_image_cohomology_operation_support)
                    ):
                        monomial_support = (
                            list_keys_dict_image_cohomology_operation_support[
                                monomial_index
                            ]
                        )
                        monomial_coefficient = dict_image_cohomology_operation_support[
                            monomial_support
                        ]

                        if (
                            list_list_images[i][k].generator
                            == list_cod_basis[j].generator
                        ):
                            bool_monomial_patch = False  ## TODO: warning...
                            if monomial_support in [
                                tuple([]),
                                (0,),
                            ] and cod_basis_trailing_support in [tuple([]), (0,)]:
                                bool_monomial_patch = True
                            if (
                                monomial_support == cod_basis_trailing_support
                                or bool_monomial_patch
                            ):
                                list_list_matrix[j][i] += (
                                    monomial_coefficient / cod_basis_leading_coeff
                                )  # 1, ..., p-1

        # DBG(self.OLDgetListListMatrix(list_dom_basis, list_cod_basis, list_list_images) == list_list_matrix)

        return list_list_matrix

    def convertKernelBasisToListOfVectors(
        self, sage_matrix_kernel_basis
    ):  ## TODO: suspicious
        list_kernel_generators = []

        if len(self.list_dom_basis) > 0:
            list_kernel_raw = list(sage_matrix_kernel_basis)

            for list_basis_linear_comb in list_kernel_raw:
                list_kernel_img_fixed_element = []

                for i in range(0, len(list_basis_linear_comb)):
                    if not list_basis_linear_comb[i] == 0:
                        list_kernel_img_fixed_element.append(
                            Element(
                                list_basis_linear_comb[i]
                                * self.list_dom_basis[i].cohomology_operation,
                                self.list_dom_basis[i].generator,
                            )
                        )

                list_kernel_generators.append(list_kernel_img_fixed_element)

        return list_kernel_generators

    def getKernelAsVect(self):
        return self.convertKernelBasisToListOfVectors(
            self.matrix.right_kernel().basis()
        )

    def convertDomVectorToLinearComb(self, list_list_matrix_vector):
        list_output_linear_cl = []

        if len(self.list_dom_basis) == 0:
            return -1

        for i in range(0, len(list(list_list_matrix_vector))):
            entry = list_list_matrix_vector[i][0]

            list_output_linear_cl.append(
                Element(
                    entry * self.list_dom_basis[i].cohomology_operation,
                    self.list_dom_basis[i].generator,
                )
            )

        return list_output_linear_cl


class MinimalResolution:
    def __init__(
        self, str_name, fixed_prime, number_of_modules, number_of_relative_degrees
    ):
        self.str_name = str_name

        self.fixed_prime = fixed_prime
        self.number_of_modules = number_of_modules
        self.number_of_relative_degrees = number_of_relative_degrees

        self.A = SteenrodAlgebra(fixed_prime, basis="adem")
        self.A_unit = self.A.monomial((0,))

        self.list_list_mapping_table = []
        self.list_list_found_generators = []
        self.list_differentials = []

        self.differential = 0

        self.list_list_expanded_minimal_resolution = []

        # Finitely presented module

        self.list_module_to_resolve_relations = []
        self.list_module_to_resolve_ev_gen = []

        # Yoneda/Massey products

        self.list_lifted_maps = []  # [[params, morph], ...]

        self.list_lift_processes = []

        self.number_of_threads = NUMBER_OF_THREADS
        self.list_processes = []

        self.list_yoneda_products = []  # [ YonedaProduct, ... ]
        self.list_e_2_massey_products = []  # [ MasseyProduct, ... ], under construction

    def createModule(self, fp_module):
        list_mapping_table = []
        list_found_generators = []

        self.list_module_to_resolve_ev_gen = fp_module.callback_generators(
            self.A, self.fixed_prime
        )

        for ev_module_generator in self.list_module_to_resolve_ev_gen:
            if ev_module_generator.generator.bool_is_free:
                lifted_generator_info = ExtendedAlgebraGenerator(
                    0,
                    ev_module_generator.generator.deg,
                    ev_module_generator.generator.index,
                    ev_module_generator.generator.bool_is_free,
                    ev_module_generator.generator.str_alias,
                )

                lift_ev_module_free_generator = Element(
                    self.A_unit, lifted_generator_info
                )

                list_found_generators.append(
                    lift_ev_module_free_generator.generator
                )  # we don't need the extra structure for generators
                list_mapping_table.append(
                    MTEntry(lift_ev_module_free_generator, [ev_module_generator])
                )

        self.list_list_mapping_table.append(list_mapping_table)
        self.list_list_found_generators.append(list_found_generators)

        print_header(f"Table of generators ({self.str_name}) [JSON]", "=", False)

        list_json_generators = [
            f'\t"{generator_element.generator.str_alias.replace("\\", "\\\\").replace(" ", "_")}": {generator_element.generator.deg}'
            for generator_element in self.list_module_to_resolve_ev_gen
        ]

        print("{\n" + ",\n".join(list_json_generators) + "\n}")

        self.list_module_to_resolve_relations = fp_module.callback_relations(
            self.A, self.fixed_prime
        )

        print_header(f"Table of relations ({self.str_name}) [JSON]", "=", False)

        if len(self.list_module_to_resolve_relations) == 0:
            print(f"(There are no extra relations)")
            print()

        for relation in self.list_module_to_resolve_relations:
            bool_found = False
            for element_generator in self.list_module_to_resolve_ev_gen:
                if element_generator.generator == relation.list_sum_output[0].generator:
                    bool_found = True
            if bool_found:
                print(
                    f'"{str(relation.steenrod_operation).replace("^", "").replace("beta", "b")} {relation.module_element.generator.str_alias.replace("\\", "\\\\").replace(" ", "_")} = {relation.list_sum_output[0].generator.str_alias.replace("\\", "\\\\").replace(" ", "_")}",'
                )

        return

    def split_support(self, support):
        r = []

        len_support = len(support)
        support_prefix = (0,)

        if self.fixed_prime == 2:
            if len_support > 1:
                r = [(support[0],), support[1:]]
            elif len_support == 1:
                r = [(0,), support]
            else:
                r = [(0,), (0,)]
        else:
            if len_support == 1:
                r = [(0,), support]
            elif len_support >= 2:
                if support[-1] == 1:  # bockstein morphism
                    r = [support[:-1] + (0,), (1,)]
                else:
                    if len(support[:-2]) == 0:
                        support_prefix = (0,)
                    else:
                        support_prefix = support[:-2]
                    r = [support_prefix, (0,) + support[-2:]]
            elif len_support == 0:
                r = [support_prefix, support]

        return r

    def non_free_eval(self, list_elements):  ########## TODO: BETA
        r = []

        for element in list_elements:
            if element.cohomology_operation == 0:
                r.append([element])
                continue

            element_r = []  # zero output value when there aren't more relations

            # A = SteenrodAlgebra(p=3, basis='adem')
            # (A.P(9)*A.P(1)*A.Q(0)).support() # se lee de derecha a izquierda: partir en 0 = potencia. partir en 1 = bockstein.

            # TODO: cambiar if/else. cambiar self.A.P
            # A.monomial( (A.Q(0)*A.P(1)*A.Q(0)).leading_support())

            trailing_support = element.cohomology_operation.trailing_support()
            coh_operation_coeff = (
                element.cohomology_operation.trailing_coefficient()
            )  ######### TODO: IMPLEMENTAR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            list_splitted_support = self.split_support(trailing_support)
            support_prefix = list_splitted_support[0]
            support_coh_operation = list_splitted_support[1]

            # print('------------------------')
            # print(element.cohomology_operation)
            # print((support_prefix))
            # print((support_coh_operation))
            # print(self.A.monomial(support_prefix))
            # print(self.A.monomial(support_coh_operation))
            # print('------------------------')
            ####print('------------------------')
            ####print(f"cohomology operation: {element.cohomology_operation}")
            ####print(f"coh_operation_coeff: {coh_operation_coeff}")
            ####print(self.A.monomial((0,)))
            ####print(self.A.monomial((0,)).trailing_support())
            ####print(f"support: {trailing_support}")
            ####print(f"support prefix: {support_prefix}")
            ####print(f"trailing support: {trailing_support}")
            ####print(f"len trailing support: {len(trailing_support)}")
            ####print('------------------------')

            coh_operation = self.A.monomial(support_coh_operation)

            if (
                support_prefix == (0,)
                and len(trailing_support) > 0
                and not support_coh_operation == (0,)
            ):  ## and len(trailing_support) > 0
                ####print(f"omg: {support_coh_operation}")
                ####print(element.cohomology_operation)

                for relation in self.list_module_to_resolve_relations:
                    if (
                        relation.steenrod_operation == coh_operation
                        and relation.module_element.generator == element.generator
                    ):
                        element_r += relation.list_sum_output

                r.append(
                    [
                        Element(
                            coh_operation_coeff * element.cohomology_operation,
                            element.generator,
                        )
                        for element in self.sum(element_r)
                    ]
                )
                # print(r)
                # if not coh_operation_coeff == self.A.monomial((0,)):
                #    print(coh_operation_coeff)
                #    print(element)
                #    sys.exit()
            elif not support_prefix == (0,):
                ####print(f"evil: {trailing_support}")
                coh_operation_prefix = coh_operation_coeff * self.A.monomial(
                    support_prefix
                )
                coh_operation_to_eval = self.A.monomial(support_coh_operation)

                element_r = []
                list_list_evals = self.non_free_eval(
                    [Element(coh_operation_to_eval, element.generator)]
                )
                for list_evals in list_list_evals:
                    element_r += self.non_free_eval(
                        [
                            Element(
                                coh_operation_prefix
                                * output_element.cohomology_operation,
                                output_element.generator,
                            )
                            for output_element in list_evals
                        ]
                    )

                for i in range(0, len(element_r)):
                    element_r[i] = self.sum(element_r[i])

                r += element_r
            else:
                ####print(f"UNREACHED? {element}")
                ####print(trailing_support)
                ####print(s elf.A.monomial(trailing_support))
                r.append([element])

        return r

    @cache
    def getSteenrodAlgebraBasis(self, deg):
        return self.A.basis(deg)

    def getElementsByDeg(self, resolution_module_subindex, deg):
        return self.getElementsByRelativeDeg(
            resolution_module_subindex, deg - resolution_module_subindex
        )

    def getElementsByRelativeDeg(self, resolution_module_subindex, module_relative_deg):
        list_found_elements = []

        if resolution_module_subindex >= 0:
            abs_deg = module_relative_deg + resolution_module_subindex

            if len(self.list_list_found_generators) > resolution_module_subindex:
                for found_generators in self.list_list_found_generators[
                    resolution_module_subindex
                ]:
                    if found_generators.deg <= abs_deg:
                        for found_element in self.getSteenrodAlgebraBasis(
                            abs_deg - found_generators.deg
                        ):
                            list_found_elements.append(
                                Element(found_element, found_generators)
                            )
        else:
            for element in self.list_module_to_resolve_ev_gen:
                if element.deg() == module_relative_deg:
                    list_found_elements.append(element)

        return list_found_elements

    def sum(self, list_elements):
        list_elements_arranged = []
        list_elements_added_generators = []

        trivial_element = Element(
            self.A_unit, AlgebraGenerator(-2, 0, 0)
        )  ## fix/change

        for element_1 in list_elements:
            element_added_partially = trivial_element
            list_elements_last_checked_generator = trivial_element

            if element_1.generator in list_elements_added_generators:
                continue

            for element_2 in list_elements:
                if element_2.generator in list_elements_added_generators:
                    continue

                if element_1 == element_2:
                    first_sum_element = element_1

                list_elements_last_checked_generator = element_1.generator
                if element_added_partially == trivial_element:
                    element_added_partially = first_sum_element
                else:
                    if element_added_partially.generator == element_2.generator:
                        element_added_partially += element_2

            if element_added_partially == trivial_element:
                continue

            list_elements_arranged.append(element_added_partially)
            list_elements_added_generators.append(list_elements_last_checked_generator)

        return list_elements_arranged

    def eval(self, module_basis_element):  # this methods depends on the mapping table
        list_img_elements = []

        for list_mapping_table in self.list_list_mapping_table:
            for mt_entry in list_mapping_table:
                if mt_entry.src.generator == module_basis_element.generator:
                    parsed_list_dst = [
                        Element(
                            module_basis_element.cohomology_operation
                            * element_dst.cohomology_operation,
                            element_dst.generator,
                        )
                        for element_dst in mt_entry.list_dst
                    ]
                    list_img_elements += parsed_list_dst

        return self.sum(
            list_img_elements
        )  # move # [element_img for element_img in self.sum(list_img_elements) if not element_img.cohomology_operation == 0]

    def raw_eval(
        self, module_basis_element
    ):  # this methods depends on the mapping table
        list_img_elements = []

        for list_mapping_table in self.list_list_mapping_table:
            for mt_entry in list_mapping_table:
                if mt_entry.src == module_basis_element:
                    list_img_elements += mt_entry.list_dst

        # DBG('img test:', self.sum(list_img_elements))

        return self.sum(list_img_elements)  ######## move

    def diff(self, resolution_module_subindex, module_relative_deg):
        if resolution_module_subindex > 0:
            list_list_images = [
                self.eval(element)
                for element in self.getElementsByRelativeDeg(
                    resolution_module_subindex, module_relative_deg
                )
            ]
            list_cod_basis = self.getElementsByRelativeDeg(
                resolution_module_subindex - 1, module_relative_deg + 1
            )
        else:
            # list_list_images = [self.raw_eval(element) for element in self.getElementsByRelativeDeg(resolution_module_subindex, module_relative_deg)]
            list_raw_eval_images = [
                self.eval(element)
                for element in self.getElementsByRelativeDeg(
                    resolution_module_subindex, module_relative_deg
                )
            ]
            # print([element for element in self.getElementsByRelativeDeg(resolution_module_subindex, module_relative_deg)])
            list_list_list_images = [
                self.non_free_eval(raw_eval_image)
                for raw_eval_image in list_raw_eval_images
            ]

            list_list_images = []
            for list_list_basis_img_substituted in list_list_list_images:
                list_rearranged_sum = self.sum(
                    [
                        basis_img
                        for list_basis_img in list_list_basis_img_substituted
                        for basis_img in list_basis_img
                    ]
                )
                list_list_images.append(list_rearranged_sum)

            # print(list_raw_eval_images)
            # print(list_list_images)
            list_cod_basis = self.getElementsByRelativeDeg(
                resolution_module_subindex - 1, module_relative_deg
            )  # this degree is absolute

        list_dom_basis = self.getElementsByRelativeDeg(
            resolution_module_subindex, module_relative_deg
        )
        # list_cod_basis = self.getElementsByRelativeDeg(resolution_module_subindex-1, module_relative_deg+1)
        ######print('-----------------------------------start')
        ######print(f"dom_basis: {list_dom_basis}")
        ######print(f"cod_basis: {list_cod_basis}")
        ######print(f"list_list_images: {list_list_images}")
        ######print(self.getElementsByRelativeDeg(resolution_module_subindex, module_relative_deg))
        ########print([Element(self.A.monomial((0,)), ExtendedAlgebraGenerator(-1, 1, 1, True, f"x^{1}"))])
        ########r = self.non_free_eval([Element(self.A.monomial((0,)), ExtendedAlgebraGenerator(-1, 1, 1, True, f"x^{1}"))])
        ######print('----------------------------end')
        d = Morphism(
            self.fixed_prime,
            list_dom_basis,
            list_cod_basis,
            list_list_images,
            tuple_dom=(resolution_module_subindex, module_relative_deg),
            tuple_cod=(
                resolution_module_subindex - 1,
                module_relative_deg + 1,
            ),  # TODO: fix first column
        )
        self.list_differentials.append(d)
        list_d_kernel = d.getKernelAsVect()

        # DBG(list_d_kernel[0][0].cohomology_operation.trailing_support())
        # sys.exit()
        dim_d_kernel = len(list_d_kernel)

        if dim_d_kernel > 0 and len(list_dom_basis) > 0:
            if (
                len(
                    self.getElementsByRelativeDeg(
                        resolution_module_subindex + 1, module_relative_deg - 1
                    )
                )
                == 0
            ):  # append a new A_p_module
                if (
                    len(self.list_list_mapping_table) - 1
                    < resolution_module_subindex + 1
                ):
                    self.list_list_mapping_table.append([])
                    self.list_list_found_generators.append([])

                list_quot_ker_img = list_d_kernel
                dim_quot_ker_img = len(list_d_kernel)

            else:
                ## Quotient with image

                list_dom_higher_deg = self.getElementsByRelativeDeg(
                    resolution_module_subindex + 1, module_relative_deg - 1
                )
                list_list_images_higher_deg = [
                    self.eval(element) for element in list_dom_higher_deg
                ]

                # DBG('@@hdom:', list_dom_higher_deg, '@@hcod:', list_dom_basis, '@@himg:', list_list_images_higher_deg)
                d_higher_degree = Morphism(
                    self.fixed_prime,
                    list_dom_higher_deg,
                    list_dom_basis,
                    list_list_images_higher_deg,
                    tuple_dom=(resolution_module_subindex + 1, module_relative_deg - 1),
                    tuple_cod=(resolution_module_subindex, module_relative_deg),
                )

                # if resolution_module_subindex == 0 and module_relative_deg >= 6:
                #    print("#################")
                #    DBG('MATRIX KERNEL', list_d_kernel)
                #    #DBG('HIGHER MATRIX IMAGE', d_higher_degree.matrix.image().basis())
                #    #DBG('HIGHER DEG IMAGES: ', list_list_images_higher_deg, 'HIGHER DEG DOM: ', list_dom_higher_deg, 'HIGHER DEG COD: ', list_dom_basis)
                #    DBG('MATRIX', d.matrix)
                #    DBG('HIGHER MATRIX', d_higher_degree.matrix)
                #    DBG('MATRIX KERNEL', d.matrix.right_kernel())
                #    #DBG('MATRIX KERNEL', d.matrix.right_kernel().basis())
                #    DBG('HIGHER MATRIX IMAGE', d_higher_degree.matrix.column_space())
                #    DBG('HIGHER DEG IMAGES: ', list_list_images_higher_deg, 'HIGHER DEG DOM: ', list_dom_higher_deg, 'HIGHER DEG COD: ', list_dom_basis)
                #    DBG('MATRIX IMAGES: ', list_list_images, 'MATRIX DOM: ', list_dom_basis, 'MATRIX COD: ', list_cod_basis)
                #    print('#'*100)
                #    print('#'*100)
                #    print('#'*100)
                #    print("TODO: checkear imagenes, armar matrices, contrastar ker...")
                #    print('#'*100)
                #    print('#'*100)
                #    print('#'*100)
                #    #printl(self.list_list_mapping_table)

                if d_higher_degree.matrix.column_space().dimension() > 0:
                    quot_ker_img = d.matrix.right_kernel().quotient(
                        d_higher_degree.matrix.column_space()
                    )
                else:
                    quot_ker_img = d.matrix.right_kernel()

                list_quot_ker_img_orig = list(
                    [item for item in quot_ker_img.basis() if not item == 0]
                )
                list_quot_ker_img = list(
                    [
                        quot_ker_img.lift(item)
                        for item in quot_ker_img.basis()
                        if not item == 0
                    ]
                )
                dim_quot_ker_img = len(list_quot_ker_img)

                list_quot_ker_img = d.convertKernelBasisToListOfVectors(
                    list_quot_ker_img
                )

                # DBG('QUOTIENT: ', quot_ker_img.basis())

                # DBG(d.matrix)
                # DBG(list_d_kernel)

                # DBG('img:', list_list_images)
                # DBG('basis:', list_dom_basis)
                # DBG('cod_basis:', list_cod_basis)
                # print('#'*100)

            for i in range(0, dim_quot_ker_img):
                element_new_generator = Element(
                    self.A_unit,
                    AlgebraGenerator(
                        resolution_module_subindex + 1,
                        module_relative_deg + resolution_module_subindex,
                        i + 1,
                    ),
                )

                # if element_new_generator.generator == AlgebraGenerator(2, 4, 1):
                #    DBG('!!!!!!!!!!!!', element_new_generator)
                #    DBG(list_quot_ker_img)
                #    DBG(list_d_kernel)
                #    DBG('##########', list(span(d.matrix.right_kernel().basis(), GF(self.fixed_prime)).quotient(d_higher_degree.matrix.column_space())))
                #    DBG('########## higher col space', list(d_higher_degree.matrix.column_space()))
                #    DBG('########## kernel', list(d.matrix.right_kernel()))
                #    print("################# MAPPING INCORRECTO DE SQ^1 2,4,1")
                #    DBG('MATRIX', d.matrix)
                #    DBG('HIGHER MATRIX', d_higher_degree.matrix)
                #    DBG('MATRIX KERNEL', d.matrix.right_kernel().basis())
                #    DBG('HIGHER MATRIX IMAGE', d_higher_degree.matrix.column_space().basis())
                #    DBG('HIGHER MATRIX KERNEL', d_higher_degree.matrix.right_kernel().basis())
                #    DBG('HIGHER DEG IMAGES: ', list_list_images_higher_deg, 'HIGHER DEG DOM: ', list_dom_higher_deg, 'HIGHER DEG COD: ', list_dom_basis)
                #    printl(self.list_list_mapping_table)
                #    sys.exit()

                self.list_list_mapping_table[resolution_module_subindex + 1].append(
                    MTEntry(element_new_generator, self.sum(list_quot_ker_img[i]))
                )

                self.list_list_found_generators[resolution_module_subindex + 1].append(
                    element_new_generator.generator
                )

                ###########print(f"[+] New generator: [{element_new_generator}] @ F_{"{"+f"{resolution_module_subindex+1}"+"}"} (dim ker: {dim_quot_ker_img}, deg: {module_relative_deg + resolution_module_subindex}).")
                # printl(self.list_list_mapping_table)

                # if dim_quot_ker_img > 1:
                #    print(list_quot_ker_img)
                #    print('-------')
                #    print(list_quot_ker_img_orig)

        self.differential = GradedMorphism(self.list_differentials)
        return

    def construct(self):
        for resolution_module_subindex in range(0, MAX_NUMBER_OF_MODULES):
            print("#" * 120)
            # print(f"[*] Current module: F_{"{"+f"{resolution_module_subindex}"+"}"}. Constructing F_{"{"+f"{resolution_module_subindex+1}"+"}"}.")

            for module_relative_deg in range(
                0, MAX_NUMBER_OF_RELATIVE_DEGREES - resolution_module_subindex
            ):  # MAX_NUMBER_OF_RELATIVE_DEGREES (t-s)
                self.diff(resolution_module_subindex, module_relative_deg)

            print(
                f"[*] Computing minimal resolution until relative degree: {MAX_NUMBER_OF_RELATIVE_DEGREES - resolution_module_subindex - 1}."
            )

            print("#" * 120)

    def expand(self):
        self.list_list_expanded_minimal_resolution = []

        for resolution_module_subindex in range(0, MAX_NUMBER_OF_MODULES):
            list_generators_tmp = []

            for module_relative_deg in range(
                0, MAX_NUMBER_OF_RELATIVE_DEGREES - resolution_module_subindex
            ):
                list_generators_tmp.append(
                    self.getElementsByRelativeDeg(
                        resolution_module_subindex, module_relative_deg
                    )
                )

            self.list_list_expanded_minimal_resolution.append(list_generators_tmp)

            return self.list_list_expanded_minimal_resolution

    def print(self):
        print("#" * 120)
        print("\t\t\t\t\tMINIMAL RESOLUTION:")
        print("#" * 120)

        if len(self.list_list_expanded_minimal_resolution) == 0:
            self.expand()

        for module_index in range(0, len(self.list_list_expanded_minimal_resolution)):
            print("#" * 120)
            ######print(f"[*] Generators of F_{"{"+f"{module_index}"+"}"}:")
            print("#" * 120)
            for relative_deg in range(
                0, len(self.list_list_expanded_minimal_resolution[module_index])
            ):
                print(
                    self.list_list_expanded_minimal_resolution[module_index][
                        relative_deg
                    ]
                )
                for element in self.list_list_expanded_minimal_resolution[module_index][
                    relative_deg
                ]:
                    print(f"[+] Images: {self.eval(element)}.")

    def toJson(self):
        list_list_min_res = []

        if len(self.list_list_expanded_minimal_resolution) == 0:
            self.expand()

        for module_index in range(0, len(self.list_list_expanded_minimal_resolution)):
            list_list_module = []

            for relative_deg in range(
                0, len(self.list_list_expanded_minimal_resolution[module_index])
            ):
                list_images = []

                for element in self.list_list_expanded_minimal_resolution[module_index][
                    relative_deg
                ]:
                    list_images.append(self.eval(element))

                list_list_module.append(
                    {
                        "list_ev_generators": [
                            str(item)
                            for item in self.list_list_expanded_minimal_resolution[
                                module_index
                            ][relative_deg]
                        ],
                        "list_images": [f"{img_item}" for img_item in list_images],
                    }
                )

                ### for k in self.list_list_expanded_minimal_resolution[module_index][relative_deg]: print(type(k))

            list_list_min_res.append(list_list_module)

        return jsons.dumps(list_list_min_res, indent=4)

    def E2chartToJson(self):
        list_chart = []

        for list_found_generators in self.list_list_found_generators:
            for generator in list_found_generators:
                list_chart.append(
                    {
                        "x": str(generator.module_index),
                        "y": str(
                            generator.deg
                            - generator.module_index
                            + (generator.index - 1) * UI_SHIFT_MULTIPLE_GENERATORS
                        ),
                        "val": str(generator),
                    }
                )

        return jsons.dumps(list_chart, indent=4)

    def create_free_graded_morphism(
        self,
        fp_map,
        callback_get_domain,
        callback_get_codomain,
        dom_module_index,
        cod_module_index,
    ):  ## TODO: BUG: varias relaciones
        list_morphism = []
        # DBG('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@', fp_map.list_tuple_domain[0], fp_map.list_tuple_codomain[0])
        # DBG(fp_map) ######### fix zero map
        if len(fp_map.list_tuple_domain) == 0:
            dom_rel_deg = 0
            cod_rel_deg = 0
        else:
            dom_rel_deg = fp_map.list_tuple_domain[0][1]
            cod_rel_deg = fp_map.list_tuple_codomain[0][1]

        for i in range(
            0, MAX_NUMBER_OF_RELATIVE_DEGREES - dom_rel_deg
        ):  ## TODO: optimize
            list_list_ordered_img = []

            list_dom_basis = callback_get_domain(dom_module_index, dom_rel_deg + i)

            for base_element in list_dom_basis:
                list_ordered_img = []

                for relation_index in range(0, len(fp_map.list_domain_generators)):
                    if (
                        base_element.generator
                        == fp_map.list_domain_generators[relation_index].generator
                    ):
                        list_ordered_img += [
                            Element(
                                base_element.cohomology_operation
                                * img.cohomology_operation,
                                img.generator,
                            )
                            for img in fp_map.list_list_images[relation_index]
                        ]

                list_list_ordered_img.append(list_ordered_img)

            list_cod_basis = callback_get_codomain(cod_module_index, cod_rel_deg + i)

            list_morphism.append(
                Morphism(
                    self.fixed_prime,
                    list_dom_basis,
                    list_cod_basis,
                    list_list_ordered_img,
                    tuple_dom=(dom_module_index, dom_rel_deg + i),
                    tuple_cod=(cod_module_index, cod_rel_deg + i),
                )
            )
        # DBG(GradedMorphism(list_morphism))

        return GradedMorphism(list_morphism)

    def lift_test(self, external_resolution):
        return self.lift_cochain(
            external_resolution,
            Element(self.A_unit, ExtendedAlgebraGenerator(1, 4, 1, False, "h_2")),
            max_cod_module_index=2,
        )

    def retrieve_lift_cochain(
        self, external_resolution, map_gen_to_lift, max_cod_module_index
    ):
        for i in range(0, len(self.list_lifted_maps)):
            lifted_map_tuple = self.list_lifted_maps[i][0]

            if (
                lifted_map_tuple[1:2] == [external_resolution, map_gen_to_lift][1:2]
            ):  # TODO: implement a hash function to compare these resolutions
                if max_cod_module_index + 1 <= lifted_map_tuple[2]:
                    if lifted_map_tuple[3] == 0:  # Interpreted as redundant
                        continue
                    return lifted_map_tuple[3][: max_cod_module_index + 1]

        return -1

    def lift_cochain(
        self,
        test,
        external_resolution,
        map_gen_to_lift,
        max_cod_module_index,
        list_list_output,
        list_index,
        mgr_lock,
    ):
        if (
            map_gen_to_lift.generator.module_index == map_gen_to_lift.generator.deg
            and map_gen_to_lift.generator.deg > 1
        ):
            list_list_output[list_index] = [
                [external_resolution, map_gen_to_lift, max_cod_module_index, 0]
            ]
            return

        dom_module_index = map_gen_to_lift.generator.module_index
        cod_module_index = 0

        first_generator_sphr = external_resolution.list_list_found_generators[0][0]

        list_lifted_map = []

        list_lifted_map.append(
            self.create_free_graded_morphism(
                FPMap(
                    [map_gen_to_lift],
                    [[Element(self.A_unit, first_generator_sphr)]],
                    [
                        (
                            map_gen_to_lift.generator.module_index,
                            map_gen_to_lift.generator.deg
                            - map_gen_to_lift.generator.module_index,
                        )
                    ],
                    [
                        (
                            first_generator_sphr.module_index,
                            first_generator_sphr.deg
                            - first_generator_sphr.module_index,
                        )
                    ],
                ),
                self.getElementsByRelativeDeg,
                external_resolution.getElementsByRelativeDeg,
                map_gen_to_lift.generator.module_index,
                first_generator_sphr.module_index,
            )
        )

        for module_index_shift in range(1, max_cod_module_index + 1):
            list_el_gen = []
            list_list_el_dst = []
            list_tuple_relative_dom = []
            list_tuple_relative_cod = []

            last_lifted_map = list_lifted_map[-1]

            print(f"[+] Lift status: ({map_gen_to_lift}):{module_index_shift}.")

            if (
                len(self.list_list_found_generators)
                < dom_module_index + module_index_shift + 1
            ):
                continue

            for found_generator in self.list_list_found_generators[
                dom_module_index + module_index_shift
            ]:
                vector_img, relative_codomain = self.differential.eval_element(
                    Element(self.A_unit, found_generator),
                    (
                        found_generator.module_index,
                        found_generator.deg - found_generator.module_index,
                    ),
                )

                vector_img, relative_codomain = last_lifted_map.eval_vector(
                    vector_img, relative_codomain
                )

                if relative_codomain == -1 or relative_codomain == (0, 0):
                    continue  ## assumed as zero

                sphere_resol_diff = (
                    external_resolution.differential.get_morphism_from_bigraded_domain(
                        (relative_codomain[0] + 1, relative_codomain[1] - 1)
                    )
                )

                if sphere_resol_diff == -1:
                    continue  ## assumed as zero

                try:
                    vector_img = sphere_resol_diff.matrix.solve_right(vector_img)
                    list_img_linear_comb = (
                        sphere_resol_diff.convertDomVectorToLinearComb(vector_img)
                    )
                    if list_img_linear_comb == -1:
                        continue

                    list_img_linear_comb = self.sum(list_img_linear_comb)
                except ValueError as e:
                    vector_img = []
                ########if map_gen_to_lift.generator == AlgebraGenerator(1, 4, 1): # and max_cod_module_index == 12:
                ########    DBG(f'found_generator (shift {module_index_shift})', Element(self.A.monomial((0,)), found_generator))
                ########    DBG('output',list_img_linear_comb)
                # DBG(Element(self.A.monomial((0,)), found_generator))
                # DBG('output',list_img_linear_comb)

                list_el_gen.append(Element(self.A_unit, found_generator))
                list_list_el_dst.append(list_img_linear_comb)
                list_tuple_relative_dom.append(
                    (
                        found_generator.module_index,
                        found_generator.deg - found_generator.module_index,
                    )
                )
                list_tuple_relative_cod.append(
                    (relative_codomain[0] + 1, relative_codomain[1] - 1)
                )

            bool_empty_morphism = False
            if len(list_el_gen) == 0:
                print(f"[+] Empty morphism ({map_gen_to_lift}), stopping computations.")
                bool_empty_morphism = True

            list_lifted_map.append(
                self.create_free_graded_morphism(
                    FPMap(
                        list_el_gen,
                        list_list_el_dst,
                        list_tuple_relative_dom,
                        list_tuple_relative_cod,
                    ),
                    self.getElementsByRelativeDeg,
                    external_resolution.getElementsByRelativeDeg,
                    dom_module_index + module_index_shift,
                    module_index_shift,
                )
            )

            print(f"[+] ({map_gen_to_lift}) lifted (until deg: {module_index_shift}).")

            if bool_empty_morphism:
                break

        # if map_gen_to_lift.generator == AlgebraGenerator(1,4,1):
        #    for i in range(0, len(list_lifted_map[1].list_morphisms)):
        #        print(list_lifted_map[1].list_morphisms[i])
        #        print(list_lifted_map[1].list_morphisms[i].list_dom_basis)
        #        print(list_lifted_map[1].list_morphisms[i].list_list_images)
        #        ## list_lifted_map[1].list_morphisms[i].list_list_images corresponds to the Yoneda product
        # DBG(list_lifted_map)

        with mgr_lock:
            list_list_output[list_index] = [
                [0, map_gen_to_lift, max_cod_module_index, list_lifted_map]
            ]  # The first entry should be some sort of hash
            # [[external_resolution, map_gen_to_lift, max_cod_module_index, list_lifted_map]]

        # DBG(list_list_output)

        # return list_lifted_map
        return

    def multiprocess_cochain_lift(
        self, external_resolution, callback_generator_filter, callback_max_module_index
    ):
        with multiprocessing.Manager() as manager:
            rearranged_found_generators = [
                found_generator
                for list_found_generators in self.list_list_found_generators
                for found_generator in list_found_generators
                if callback_generator_filter(found_generator)
            ]

            len_output_list = len(rearranged_found_generators)
            manager_lock = manager.Lock()
            list_list_output = manager.list([[]] * len_output_list)

            print(f"Creating {len_output_list} subprocesses...")

            for generator_i in range(0, len(rearranged_found_generators)):
                generator_to_lift = rearranged_found_generators[generator_i]
                element_generator_to_lift = Element(self.A_unit, generator_to_lift)

                self.list_lift_processes.append(
                    multiprocessing.Process(
                        target=self.lift_cochain,
                        args=(
                            self,
                            external_resolution,
                            element_generator_to_lift,
                            callback_max_module_index(generator_to_lift),
                            list_list_output,
                            generator_i,
                            manager_lock,
                        ),
                    )
                )

                print(f"[*] Subprocess associated to generator: {generator_to_lift}")

            t = 0
            k = NUMBER_OF_THREADS
            while k <= -(len_output_list // -NUMBER_OF_THREADS) * NUMBER_OF_THREADS:
                for i in range(k - NUMBER_OF_THREADS, min(k, len_output_list)):
                    process = self.list_lift_processes[i]
                    process.start()

                for i in range(k - NUMBER_OF_THREADS, min(k, len_output_list)):
                    process = self.list_lift_processes[i]
                    process.join()
                    t += 1
                    print(
                        f"[*] {len(rearranged_found_generators) - t} subprocesses remaining."
                    )

                k += NUMBER_OF_THREADS

            self.list_lifted_maps = list(list_list_output)

            self.list_lift_processes = []

        return

    def retrieve_yoneda_products(
        self,
        external_resolution,
        min_module_index=0,
        max_module_index=-1,
        max_deg=sys.maxsize,
    ):
        ## test param ...
        if max_module_index == -1:
            max_module_index = len(self.list_list_found_generators)

        for external_found_generators in external_resolution.list_list_found_generators:
            for external_generator in external_found_generators:
                for internal_found_generator in self.list_list_found_generators[
                    min_module_index:max_module_index
                ]:
                    for generator_to_lift in internal_found_generator:
                        if external_generator.deg > max_deg:
                            continue
                        if (
                            external_generator.module_index == external_generator.deg
                            and external_generator.deg > 1
                        ):
                            continue  # hardcoded

                        #########if not generator_to_lift in [AlgebraGenerator(0, 18, 1), AlgebraGenerator(1, 54, 1), AlgebraGenerator(3, 20, 1)]:
                        #########    continue
                        #########if generator_to_lift == AlgebraGenerator(1, 54, 1) and external_generator.deg > 19:
                        #########    continue
                        ####if external_generator.module_index > 0:
                        ####    if (external_generator.deg - external_generator.module_index) % external_generator.module_index == 0 and external_generator.deg > 4:
                        ####        continue ####### fast fix for h_0 powers
                        # if not external_generator == AlgebraGenerator(2, 9, 1) or not generator_to_lift == AlgebraGenerator(1, 4, 1):
                        #     continue

                        print(
                            f"Computing Yoneda product: {external_generator} @@ {generator_to_lift}"
                        )

                        list_lifted_map = self.retrieve_lift_cochain(
                            external_resolution,
                            Element(self.A_unit, generator_to_lift),
                            external_generator.module_index,
                        )

                        if list_lifted_map == -1:
                            continue

                        for morphism in list_lifted_map[-1].list_morphisms:
                            list_candidates_found = []

                            # if external_generator == AlgebraGenerator(2, 12, 1) and generator_to_lift == AlgebraGenerator(0, 0, 1):
                            #    DBG('list_image', morphism.list_list_images)
                            #    DBG('list_dom_basis', morphism.list_dom_basis)
                            #    DBG('list_cod_basis', morphism.list_cod_basis)
                            #    DBG(morphism)
                            #    DBG(len(list_lifted_map[-1].list_morphisms))
                            #    #DBG(image == Element(self.A.monomial((0,)), external_generator))
                            #    DBG('external_generator', external_generator)

                            for k in range(0, len(morphism.list_list_images)):
                                list_image = morphism.list_list_images[k]

                                for i in range(0, len(list_image)):
                                    image = list_image[i]

                                    if not image.cohomology_operation.is_zero():
                                        if (
                                            image.generator == external_generator
                                            and image.cohomology_operation.degree() == 0
                                        ):
                                            list_candidates_found.append(
                                                [
                                                    morphism.list_dom_basis[k],
                                                    image.cohomology_operation.leading_coefficient(),
                                                ]
                                            )  # TODO: change format

                            if len(list_candidates_found) > 0:
                                list_linear_comb_img = [
                                    Element(
                                        self.A_unit * leading_coefficient,
                                        candidate_found.generator,
                                    )
                                    for candidate_found, leading_coefficient in list_candidates_found
                                ]
                                list_linear_comb_generators = self.sum(
                                    list_linear_comb_img
                                )
                                if len(list_linear_comb_generators) == 1:
                                    if list_linear_comb_generators[
                                        0
                                    ].cohomology_operation.is_zero():
                                        break

                                yoneda_product = YonedaProduct(
                                    external_generator,
                                    generator_to_lift,
                                    list_linear_comb_generators,
                                )

                                log(str(yoneda_product))

                                self.list_yoneda_products.append(yoneda_product)
                                break

        print("Yoneda products computed successfully.")

        # printl(self.list_yoneda_products)

        return

    def yoneda_products_to_plot_coordinates(self):
        list_output = []

        for yoneda_product in self.list_yoneda_products:
            internal_generator = yoneda_product.internal_generator
            linear_comb_external_generators = (
                yoneda_product.linear_comb_output_generators
            )

            src_x = (
                internal_generator.deg
                - internal_generator.module_index
                + UI_SHIFT_MULTIPLE_GENERATORS * (internal_generator.index - 1)
            )
            src_y = internal_generator.module_index

            for external_generator in linear_comb_external_generators:
                dst_x = (
                    external_generator.generator.deg
                    - external_generator.generator.module_index
                    + UI_SHIFT_MULTIPLE_GENERATORS
                    * (external_generator.generator.index - 1)
                )
                dst_y = external_generator.generator.module_index

                list_output.append([(src_x, src_y), (dst_x, dst_y)])

        return list_output


## MODULES

# RP^\infty (p=2)


def callback_coh_rp_infty_generators(A, prime, max_deg):
    output_list = []

    HARDCODED_STEENROD_ALG_UNIT = A.monomial((0,))

    for k in range(1, max_deg + 1):
        bool_free_module_generator = False

        if k in [2**j - 1 for j in range(0, max_deg + 1)]:  ## replace with bitwise AND
            bool_free_module_generator = True

        ev_module_generator = Element(
            HARDCODED_STEENROD_ALG_UNIT,
            ExtendedAlgebraGenerator(-1, k, 1, bool_free_module_generator, f"x^{k}"),
        )

        output_list.append(ev_module_generator)

    return output_list


def callback_coh_rp_infty_relations(A, prime, max_deg):
    output_list = []

    HARDCODED_STEENROD_ALG_UNIT = A.monomial((0,))

    for k in range(1, max_deg + 1):
        for i in range(1, k + 1):  # skip identity relation
            if k in [
                2**j - 1 for j in range(0, max_deg + 1)
            ]:  ## replace with bitwise AND
                bool_free_module_generator = True

            output_list.append(
                AlgebraGeneratorRelation(
                    Element(
                        HARDCODED_STEENROD_ALG_UNIT,
                        ExtendedAlgebraGenerator(
                            -1, k, 1, bool_free_module_generator, f"x^{k}"
                        ),
                    ),
                    A.P(i),
                    [
                        Element(
                            bin_coeff(k, i) * HARDCODED_STEENROD_ALG_UNIT,
                            ExtendedAlgebraGenerator(
                                -1, k + i, 1, bool_free_module_generator, f"x^{k+i}"
                            ),
                        ),
                    ],
                )
            )

    return output_list


# S^0


def callback_coh_sphere_generators(A, prime, max_deg):
    return [
        Element(A.monomial((0,)), ExtendedAlgebraGenerator(-1, 0, 1, True, f"x_{0}"))
    ]


def callback_coh_sphere_relations(A, prime, max_deg):
    return []


# S^\rho_{D_3} (\rho given by Samuel's article) [p: odd]


def callback_coh_p_odd_hom_orbit_representation_sphere_rho_d_3_generators(
    A, prime, max_deg
):
    @cache
    def deg2tuple(deg):
        if deg == connectedness:
            return (0, 0)  # unused, u \notin J_1
        i, j = (0, 0)
        if (deg - connectedness) % 4 == 1:
            i = 1
        if (deg - connectedness) % 4 <= 2 and (deg - connectedness) % 4 > 0:
            j = (deg - connectedness - i) // 2  # |s| = 2
        return (i, j)

    output_list = []

    HARDCODED_STEENROD_ALG_UNIT = A.monomial((0,))

    connectedness = 3  # |u| = 3

    # ev_module_generator = Element(A.monomial((0,)), ExtendedAlgebraGenerator(-1, connectedness, 1, True, f"u"))

    for k in range(connectedness + 1, connectedness + max_deg):
        bool_free_module_generator = True

        # TODO: freeness criterion
        bool_free_module_generator = False
        if k in [4, 12, 36, 108]:
            bool_free_module_generator = True

        i, j = deg2tuple(k)
        if (i, j) == (0, 0):
            continue

        ev_module_generator = Element(
            HARDCODED_STEENROD_ALG_UNIT,
            ExtendedAlgebraGenerator(
                -1, k, 1, bool_free_module_generator, f"u \\alpha^{i} s^{j}"
            ),
        )

        output_list.append(ev_module_generator)

    return output_list


def callback_coh_p_odd_hom_orbit_representation_sphere_rho_d_3_relations(
    A, prime, max_deg
):
    connectedness = 3  # |u| = 3, redundant

    @cache
    def deg2tuple(deg):
        if deg == connectedness:
            return (0, 0)  # unused, u \notin J_1
        i, j = (0, 0)
        if (deg - connectedness) % 4 == 1:
            i = 1
        if (deg - connectedness) % 4 <= 2 and (deg - connectedness) % 4 > 0:
            j = (deg - connectedness - i) // 2  # |s| = 2
        return (i, j)

    def bockstein_coeff(tuple_element_exponents):
        i, j = tuple_element_exponents
        int_coeff = 0
        if not i == 0:
            int_coeff = -1
        return int_coeff  # "j" just affect the grading

    @cache
    def power_coeff(tuple_element_exponents, k):  # P^k
        i, j = tuple_element_exponents
        return sum(
            [bin_coeff(3 - 2, r) * bin_coeff(j, k - r) for r in range(0, k + 1)]
        )  # the grading effect is hardcoded

    output_list = []

    HARDCODED_STEENROD_ALG_UNIT = A.monomial((0,))

    for int_deg in range(connectedness + 1, connectedness + max_deg):
        i, j = deg2tuple(int_deg)

        if (i, j) == (0, 0):
            continue

        c_bockstein = bockstein_coeff((i, j))

        ## TODO: bool_free_module_generator
        bool_free_module_generator = False
        if int_deg in [4, 12, 36, 108]:
            bool_free_module_generator = True

        bool_free_module_generator_img = False
        if int_deg + 1 in [4, 12, 36, 108]:
            bool_free_module_generator_img = True

        current_element = Element(
            HARDCODED_STEENROD_ALG_UNIT,
            ExtendedAlgebraGenerator(
                -1, int_deg, 1, bool_free_module_generator, f"u \\alpha^{i} s^{j}"
            ),
        )

        output_list.append(
            AlgebraGeneratorRelation(
                current_element,
                A.monomial((1,)),
                [
                    Element(
                        c_bockstein * HARDCODED_STEENROD_ALG_UNIT,
                        ExtendedAlgebraGenerator(
                            -1,
                            int_deg + 1,
                            1,
                            bool_free_module_generator_img,
                            f"u \\alpha^{i-1} s^{j+1}",
                        ),
                    ),
                ],  # 0*...
            )
        )

        for k in range(1, int_deg + 1):
            bool_free_module_generator_img = False
            if int_deg + k in [4, 12, 36, 108]:
                bool_free_module_generator_img = True

            c_power = power_coeff((i, j), k)
            output_list.append(
                AlgebraGeneratorRelation(
                    current_element,
                    A.monomial((0, k, 0)),
                    [
                        Element(
                            c_power * HARDCODED_STEENROD_ALG_UNIT,
                            ExtendedAlgebraGenerator(
                                -1,
                                int_deg + 4 * k,
                                1,
                                bool_free_module_generator_img,
                                f"u \\alpha^{i} s^{j+2*k}",
                            ),
                        ),
                    ],
                )
            )

    l = []
    for item in output_list:
        if (
            item.list_sum_output[0].cohomology_operation == HARDCODED_STEENROD_ALG_UNIT
            or item.list_sum_output[0].cohomology_operation
            == 2 * HARDCODED_STEENROD_ALG_UNIT
        ):
            l.append(item.list_sum_output[0].generator.deg)
            print(item.list_sum_output[0].generator)
    print("-" * 120)
    print("TODO: rearrange this part.")
    for t in [
        k if k not in sorted(l) and (not k % 4 == 2 and not k % 4 == 3) else -1
        for k in range(4, MAX_NUMBER_OF_RELATIVE_DEGREES)
    ]:
        if not t == -1:
            print(t)
    # sys.exit()
    print("-" * 120)

    return output_list


# S^\rho_{D_3} (\rho given by Samuel's article) [p=even]


def callback_coh_p_even_hom_orbit_representation_sphere_rho_d_3_generators(
    A, prime, max_deg
):
    @cache
    def deg2tuple(deg):
        if deg == connectedness:
            return (0, 0)  # unused, u \notin J_1
        i, j = (0, 0)
        if (deg - connectedness) % 4 == 1 or (deg - connectedness) % 4 == 2:
            i = -1
            j = -1
        if (deg - connectedness) % 4 == 3:
            i = 1
            j = (deg - connectedness) // 2
        if (deg - connectedness) % 4 == 0:
            j = (deg - connectedness) // 2  # |s| = 2
        return (i, j)

    output_list = []

    HARDCODED_STEENROD_ALG_UNIT = A.monomial((0,))

    connectedness = 3  # |u| = 3

    for k in range(connectedness, connectedness + max_deg + 1):
        bool_free_module_generator = True

        # TODO: freeness criterion
        bool_free_module_generator = False
        if k in [3, 6, 18, 54]:
            bool_free_module_generator = True

        i, j = deg2tuple(k)
        if (i, j) == (-1, -1):
            continue

        ev_module_generator = Element(
            HARDCODED_STEENROD_ALG_UNIT,
            ExtendedAlgebraGenerator(
                -1, k, 1, bool_free_module_generator, f"u \\alpha^{i} s^{j}"
            ),
        )

        output_list.append(ev_module_generator)

    return output_list


def callback_coh_p_even_hom_orbit_representation_sphere_rho_d_3_relations(
    A, prime, max_deg
):
    connectedness = 3  # |u| = 3, redundant

    @cache
    def deg2tuple(deg):
        if deg == connectedness:
            return (0, 0)  # unused, u \notin J_1
        i, j = (0, 0)
        if (deg - connectedness) % 4 == 1 or (deg - connectedness) % 4 == 2:
            i = -1
            j = -1
        if (deg - connectedness) % 4 == 3:
            i = 1
            j = (deg - connectedness) // 2
        if (deg - connectedness) % 4 == 0:
            j = (deg - connectedness) // 2  # |s| = 2
        return (i, j)

    def bockstein_coeff(tuple_element_exponents):
        i, j = tuple_element_exponents
        int_coeff = 0
        if not i == 0:
            int_coeff = -1
        return int_coeff  # "j" just affect the grading

    @cache
    def power_coeff(tuple_element_exponents, k):  # P^k
        i, j = tuple_element_exponents
        return sum(
            [bin_coeff(3 - 2, r) * bin_coeff(j, k - r) for r in range(0, k + 1)]
        )  # the grading effect is hardcoded

    output_list = []

    HARDCODED_STEENROD_ALG_UNIT = A.monomial((0,))

    for int_deg in range(connectedness, connectedness + max_deg + 1):
        i, j = deg2tuple(int_deg)

        if (i, j) == (-1, -1):
            continue

        c_bockstein = bockstein_coeff((i, j))

        ## TODO: bool_free_module_generator
        bool_free_module_generator = False
        if int_deg in [3, 6, 18, 54]:
            bool_free_module_generator = True

        bool_free_module_generator_img = False
        if int_deg + 1 in [3, 6, 18, 54]:
            bool_free_module_generator_img = True

        current_element = Element(
            HARDCODED_STEENROD_ALG_UNIT,
            ExtendedAlgebraGenerator(
                -1, int_deg, 1, bool_free_module_generator, f"u \\alpha^{i} s^{j}"
            ),
        )

        output_list.append(
            AlgebraGeneratorRelation(
                current_element,
                A.monomial((1,)),
                [
                    Element(
                        c_bockstein * HARDCODED_STEENROD_ALG_UNIT,
                        ExtendedAlgebraGenerator(
                            -1,
                            int_deg + 1,
                            1,
                            bool_free_module_generator_img,
                            f"u \\alpha^{max(i-1, 0)} s^{j+1}",
                        ),
                    ),
                ],  # 0*...
            )
        )

        for k in range(1, int_deg + 1):
            bool_free_module_generator_img = False
            if int_deg + k in [3, 6, 18, 54]:
                bool_free_module_generator_img = True

            c_power = power_coeff((i, j), k)

            output_list.append(
                AlgebraGeneratorRelation(
                    current_element,
                    A.monomial((0, k, 0)),
                    [
                        Element(
                            c_power * HARDCODED_STEENROD_ALG_UNIT,
                            ExtendedAlgebraGenerator(
                                -1,
                                int_deg + 4 * k,
                                1,
                                bool_free_module_generator_img,
                                f"u \\alpha^{i} s^{j+2*k}",
                            ),
                        ),
                    ],
                )
            )

    # printl(output_list)
    # sys.exit()
    l = []
    for item in output_list:
        if (
            item.list_sum_output[0].cohomology_operation == HARDCODED_STEENROD_ALG_UNIT
            or item.list_sum_output[0].cohomology_operation
            == 2 * HARDCODED_STEENROD_ALG_UNIT
        ):
            l.append(item.list_sum_output[0].generator.deg)
            print(item.list_sum_output[0].generator)
    print("-" * 120)
    for t in [
        k if k not in sorted(l) and (not k % 4 == 0 and not k % 4 == 1) else -1
        for k in range(3, MAX_NUMBER_OF_RELATIVE_DEGREES)
    ]:
        if not t == -1:
            print(t)
    # sys.exit()

    return output_list


##################################################################################
##                                  SET UP                                      ##
##################################################################################

print_banner()

starting_time = time.time()

str_name_sphere = "Sphere"
str_name_module_to_resolve = "Sphere"  # "S_rho_D_3__p-odd"  # "S_rho_D_3__p-even"
# str_name_module_to_resolve = "Sphere"

str_output_file_sphere = f"{FIXED_PRIME_NUMBER}-{str_name_sphere}_{MAX_NUMBER_OF_RELATIVE_DEGREES}_{DEFAULT_YONEDA_PRODUCT_MAX_DEG}"
str_output_file_module = f"{FIXED_PRIME_NUMBER}-{str_name_module_to_resolve}_{MAX_NUMBER_OF_RELATIVE_DEGREES}_{DEFAULT_YONEDA_PRODUCT_MAX_DEG}"

# Minimal resolution of the sphere (required to compute Yoneda and Massey products)

minimalResolutionSphere = load_object(str_output_file_sphere)
if not minimalResolutionSphere:
    minimalResolutionSphere = MinimalResolution(
        str_name_sphere,
        FIXED_PRIME_NUMBER,
        MAX_NUMBER_OF_MODULES,
        MAX_NUMBER_OF_RELATIVE_DEGREES,
    )

    coh_sphere_presentation = FPModule(
        "Cohomology of the sphere spectrum",
        callback_coh_sphere_generators,
        callback_coh_sphere_relations,
        0,
    )
    minimalResolutionSphere.createModule(coh_sphere_presentation)
    minimalResolutionSphere.construct()

    dump_object(minimalResolutionSphere, str_output_file_sphere)

# Module minimal resolution

minimalResolution = load_object(f"{str_output_file_module}__additive__")
if not minimalResolution:
    minimalResolution = MinimalResolution(
        str_name_module_to_resolve,
        FIXED_PRIME_NUMBER,
        MAX_NUMBER_OF_MODULES,
        MAX_NUMBER_OF_RELATIVE_DEGREES,
    )

    ####coh_sphere_presentation = FPModule(callback_coh_sphere_generators, callback_coh_sphere_relations, 0)
    ####minimalResolution.createModule(coh_sphere_presentation)
    # minimalResolution.construct()

    ##minimalResolution.lift_test(minimalResolutionSphere)
    # minimalResolution.compute_yoneda_products(minimalResolutionSphere)

    # coh_rp_infty_presentation = FPModule(callback_coh_rp_infty_generators, callback_coh_rp_infty_relations, 20) # Finitely presented module to resolve
    ##coh_sphere_presentation = FPModule(callback_coh_sphere_generators, callback_coh_sphere_relations, 0)
    ###################callback_coh_p_odd_representation_sphere_rho_d_3_presentation = FPModule(
    ###################    callback_coh_p_odd_hom_orbit_representation_sphere_rho_d_3_generators,
    ###################    callback_coh_p_odd_hom_orbit_representation_sphere_rho_d_3_relations,
    ###################    MAX_NUMBER_OF_RELATIVE_DEGREES
    ###################)
    ###################minimalResolution.createModule(callback_coh_p_odd_representation_sphere_rho_d_3_presentation) # Finitely presented module to resolve

    callback_coh_p_odd_representation_sphere_rho_d_3_presentation = FPModule(
        "Cohomology of the 3-local sphere" # "Cohomology of representation sphere S^{\\rho}_{D_3} (p: odd)",
        callback_coh_sphere_generators # callback_coh_p_odd_hom_orbit_representation_sphere_rho_d_3_generators,
        callback_coh_sphere_relations # callback_coh_p_odd_hom_orbit_representation_sphere_rho_d_3_relations,
        MAX_NUMBER_OF_RELATIVE_DEGREES,
    )
    minimalResolution.createModule(
        callback_coh_p_odd_representation_sphere_rho_d_3_presentation
    )  # Finitely presented module to resolve
    minimalResolution.construct()

    log(minimalResolution.E2chartToJson())

    dump_object(minimalResolution, f"{str_output_file_module}__additive__")

if BOOL_COMPUTE_ONLY_ADDITIVE_STRUCTURE:
    print(f"[+] BOOL_COMPUTE_ONLY_ADDITIVE_STRUCTURE: True. Exiting.")
    sys.exit()

# Chain map lifts

minimalResolution_lifts = load_object(f"{str_output_file_module}__lifts_")
if not minimalResolution_lifts:
    cbk_filter = (
        lambda x: True # if (x.module_index, x.deg - x.module_index) == (0, 18) else False
    )
    cbk_max_deg = lambda x: MAX_NUMBER_OF_MODULES - x.deg
    minimalResolution.multiprocess_cochain_lift(
        minimalResolutionSphere, cbk_filter, cbk_max_deg
    )  ## GEN_CALLBACK = lambda x: True

    minimalResolution.retrieve_yoneda_products(minimalResolutionSphere)

    dump_object(minimalResolution.list_lifted_maps, f"{str_output_file_module}__lifts_")

    print("Saved computations.")
else:
    minimalResolution.list_lifted_maps = minimalResolution_lifts

    minimalResolution.retrieve_yoneda_products(minimalResolutionSphere)

##################################################################################
##                                    DRAW                                      ##
##################################################################################


def check_drawable_segment(
    list_current_coordinates, list_list_found_generators, x_shift, y_shift
):
    bool_found_nontrivial_codomain = False

    element_module_index, element_deg = list_current_coordinates

    for module_index_2 in range(0, len(list_list_found_generators)):
        for relative_deg_2 in range(0, len(list_list_found_generators[module_index_2])):
            element_module_index_2 = list_list_found_generators[module_index_2][
                relative_deg_2
            ].module_index
            element_deg_2 = list_list_found_generators[module_index_2][
                relative_deg_2
            ].deg
            element_index_2 = list_list_found_generators[module_index_2][
                relative_deg_2
            ].index

            if (
                element_module_index_2 == element_module_index + y_shift
                and (element_deg_2 - module_index_2)
                == element_deg - module_index + x_shift
            ):
                bool_found_nontrivial_codomain = True

    return bool_found_nontrivial_codomain


# minimalResolution.print()
# print(minimalResolution.toJson())
print(minimalResolution.E2chartToJson())

# printl(minimalResolution.yoneda_products_to_plot_coordinates())

fig = go.Figure()

dict_tuples_yoneda_products = {}

group_x = []
group_y = []

for module_index in range(0, len(minimalResolution.list_list_found_generators)):
    for relative_deg in range(
        0, len(minimalResolution.list_list_found_generators[module_index])
    ):
        element_module_index = minimalResolution.list_list_found_generators[
            module_index
        ][relative_deg].module_index
        element_deg = minimalResolution.list_list_found_generators[module_index][
            relative_deg
        ].deg
        element_index = minimalResolution.list_list_found_generators[module_index][
            relative_deg
        ].index

        group_x.append(
            element_deg
            - module_index
            + (element_index - 1) * UI_SHIFT_MULTIPLE_GENERATORS
        )
        group_y.append(element_module_index)

        list_line_styles = ["--", ":", "-.", "-", "-", "-", "-"]
        for num_page in range(2, 6):
            if check_drawable_segment(
                [element_module_index, element_deg],
                minimalResolution.list_list_found_generators,
                -1,
                num_page,
            ):
                fig.add_trace(
                    go.Scatter(
                        x=[
                            element_deg
                            - element_module_index
                            + (element_index - 1) * UI_SHIFT_MULTIPLE_GENERATORS,
                            element_deg
                            - module_index
                            + (element_index - 1) * UI_SHIFT_MULTIPLE_GENERATORS
                            - 1,
                        ],
                        y=[element_module_index, element_module_index + num_page],
                        name=f"({element_deg - module_index}, {element_module_index}),({element_deg - module_index - 1}, {element_module_index + num_page})-d_{num_page}",
                        line=dict(color="red", width=0.8),
                    )
                )

        for (x0, y0), (
            x1,
            y1,
        ) in minimalResolution.yoneda_products_to_plot_coordinates():
            str_diff_key = f"{int(x1-x0)},{int(y1-y0)}"

            bool_key_found = False
            for key in dict_tuples_yoneda_products.keys():
                if key == str_diff_key:
                    bool_key_found = True  # break

            if bool_key_found:
                dict_tuples_yoneda_products[str_diff_key].append((x0, x1, y0, y1))
            else:
                dict_tuples_yoneda_products[str_diff_key] = [(x0, x1, y0, y1)]

fig.update_yaxes(range=[0, 28], maxallowed=28)

for key in dict_tuples_yoneda_products.keys():
    list_tuple_differentials = dict_tuples_yoneda_products[key]

    x_diff = []
    y_diff = []

    for tuple_differentials in list_tuple_differentials:
        x_diff += [tuple_differentials[0], tuple_differentials[1], None]
        y_diff += [tuple_differentials[2], tuple_differentials[3], None]

    fig.add_trace(
        go.Scatter(
            x=x_diff,
            y=y_diff,
            name=f"({key})-product",
            line=dict(color="#0F0", width=0.05),
        )  # .update_traces(visible='legendonly')
    )

fig.add_trace(
    go.Scatter(
        x=group_x,
        y=group_y,
        mode="markers",
        name="Copy of F_3",
        marker=dict(size=5, color="#0F0"),
    )
)

fig.update_layout(
    template="plotly_dark",
    legend_traceorder="reversed",
    xaxis=dict(autorange=True, fixedrange=False),
    yaxis=dict(autorange=True, fixedrange=False),
    # updatemenus=[
    #    dict(
    #     type='buttons',
    #     buttons=list([
    #        dict(label = 'Show cluster 1',
    #             method = 'update',
    #             args = [{'visible': [True, False]},
    #                     {'title': 'cluster 1'}]),
    #
    #        dict(label = 'Show cluster 2',
    #             method = 'update',
    #             args = [{'visible': [False, True]},
    #                     {'title': 'cluster 2'}]),
    #
    #        dict(label = 'Show both clusters',
    #             method = 'update',
    #             args = [{'visible': [True, True]},
    #                     {'title': 'both'}])
    #    ]),
    # )]
)

fig.update_xaxes(range=[0, 10])
fig.update_yaxes(range=[0, 10])
fig.write_html("./chart.html")

print(f"[+] Process finished. Elapsed time: {time.time() - starting_time} (s).")

fig.show()
