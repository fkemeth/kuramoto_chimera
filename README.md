INSTALLATION
---------

Via pip:

`(sudo) pip install kuramoto_chimera`

Via source

(https://github.com/fkemeth/kurmoto_chimera)

DOCUMENTATION
---------

This python package contains functions to integrate
a system of nonlocally-coupled phase oscillators
investigated by Y. Kuramoto and D. Battogtokh,
as described in their paper

"Coexistence of Coherence and Incoherence in Nonlocally Coupled Phase Oscillators."
(http://www.j-npcs.org/abstracts/vol2002/v5no4/v5no4p380.html)

To integrate the system with the predefined parameters, run


    from kuramoto_chimera import integrate
    Ad = integrate()

To plot the last snapshot, run

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(Ad["xx"], Ad["data"][-1])
    plt.show()


ISSUES
---------

For questions, please contact (<felix@kemeth.de>), or visit [the GitHub repo](https://github.com/fkemeth/kuramoto_chimera).


LICENCE
---------


This work is licenced under GNU General Public License v3.
This means you must cite

"A classification scheme for chimera states"
F.P. Kemeth et al.
(http://dx.doi.org/10.1063/1.4959804)

if you use this package for publications.
