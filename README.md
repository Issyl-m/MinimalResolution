# MinimalResolution [proof-of-concept]

Proof-of-concept of computing minimal resolutions over the mod $p$ Steenrod algebra developed by me. This program constructs a free resolution of a finitely presented $`\mathbb{F}_p-`$module as an $\mathcal{A}_p-$module. It also computes chain map lifts of cohomology classes to compute Yoneda products.

# Requirements 

This software should be executed on ``GNU/Linux`` distributions.

1. Install ``Python``.
2. Install ``SageMath``.
3. Run ``pip install plotly``.
4. Run ``pip install jsons``.

# Usage (Linux users)

1. Provide a finitely presented $\mathbb{F}_p-$module by using the ``minimalResolution.createModule()`` method. The source code provides some examples.
2. Execute `g.sh` (or ``python -W ignore minrv1.py``).

You will get a plot resembling the computed `Ext groups`, and the corresponding $`E_2-`$term with its Yoneda products will be saved in ``./log.txt`` for further inspection. The program also dumps the computed lifts and the minimal resolution into ``./*.obj`` files.

Note that we are not only interested in these `Ext groups` and the associated ``Yoneda products``. The program also stores a backup of the objects created by the software, including chain lift information.

# Usage (non-Linux users)

Alternatively, you can run this program on ``Google Colab``. Just copy and paste the content of ``./colab/i.sh``. You will also need to hardcode a finitely presented module and then call ``minimalResolution.createModule()``.

# Parameters

* ``BOOL_COMPUTE_ONLY_ADDITIVE_STRUCTURE`` 
  - This parameter indicates whether the software should compute lifts or not.

* ``FIXED_PRIME_NUMBER`` 
  - This is the associated prime number used during the execution.
  - You must provide a prime number compatible with the finitely presented $\mathbb{F}_p-$module to consider.

* ``MAX_NUMBER_OF_RELATIVE_DEGREES`` 
  - This parameter accounts for the maximum relative (topological) degree to compute.

* ``MAX_NUMBER_OF_MODULES`` 
  - Accounts for the maximum $\mathcal{A}_p-$modules to compute in the minimal resolution.

* ``NUMBER_OF_THREAS``
