.. jaxaHackathon documentation master file, created by
   sphinx-quickstart on Mon Jun 26 18:00:15 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=====================
OpenVino Introduction
=====================

PYNQ is an open-source project from AMD. It provides a Jupyter-based
framework with Python APIs for using AMD Xilinx Adaptive Computing platforms.
PYNQ supports ZynqR and Zynq Ultrascale+?, Zynq RFSoC?, Kria? SOMs, Alveo?
and AWS-F1 instances.

PYNQ enables architects, engineers
and programmers who design embedded systems to use Adaptive Computing
platforms, without having
to use ASIC-style design tools to design programmable logic circuits.


PYNQ Background
===============

* Programmable logic circuits are presented as hardware libraries called
  *overlays*.  These overlays are analogous to software libraries.  A software
  engineer can select the overlay that best matches their application.  The
  overlay can be accessed through an Python API. Creating a new overlay still 
  requires engineers with expertise in designing programmable logic circuits.  
  The key difference however, is the
  *build once, re-use many times* paradigm. Overlays, like software libraries,
  are designed to be configurable and re-used as often as possible in many
  different applications.


.. toctree::
   :maxdepth: 5
   :caption: Contents:
   :numbered:

   ./getting_started
   ./performance_evaluation


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
