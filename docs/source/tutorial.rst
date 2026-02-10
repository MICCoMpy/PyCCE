Tutorials
===================

The examples below are available as Jupyter notebooks in the GitHub repository.

.. toctree::
   :maxdepth: 1
   :caption: Examples of using PyCCE

   tutorials/diamond_nv
   tutorials/sic_vv
   tutorials/si_shallow
   tutorials/classical_noise
   tutorials/second_spin
   tutorials/mecce


The recommended order of the tutorials is:

* :doc:`tutorials/diamond_nv` goes through the :doc:`quickstart` example in more details.
* :doc:`tutorials/sic_vv` explores the difference between
  generalized CCE with and without random bath state sampling.
  Also, in this example we introduce the way to work with hyperfine tensors computed from DFT.
* :doc:`tutorials/si_shallow` shows how to include custom hyperfine couplings for more
  delocalized defects in semiconductors.
* :doc:`tutorials/classical_noise` explains how to use autocorrelation function of the noise
  to predict the decay of the coherence of the NV center in diamond.
* :doc:`tutorials/second_spin` goes over systems with two central spins, either forming a hybrid qubit,
  or two entangled qubits.
* :doc:`tutorials/mecce` provides details on the master-eqation based solvers ME-CCE and ME-gCCE.


