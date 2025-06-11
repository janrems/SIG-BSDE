# SIG-BSDE 

This is a code that accompanies the paper titled [SIG-BSDE for Dynamic Risk Measures](https://arxiv.org/abs/2408.02853) that is a joint work with Nacira Agram and Emanuela Rosazza Gianin. Please refer to the abovementioned paper when using code on this repository. 
In particular, we develop a signature-based algorithm for solving BSDEs, which we apply to the setting of dynamical risk measures. In particular, we take a look at the dynamic risk measures under ambiguity, which we solve numerically with the help of machine learning.
## Code Structure

* **BSDE_Cex.py**: Contains the class for solving BSDEs
* **DBDP.py**: Contains the deep learning [DBDP](https://link.springer.com/article/10.1007/s42985-020-00062-8)
* **beta_BSDE.py**: Contains the BSDE and machine learning classes used for solving ambiguity problems
* **beta_BSDE2D.py**: Contains the BSDE and machine learning classes used for solving dual ambiguity problems
* The corresponding files with the suffix **_example** contain particular examples where the performance of the algorithms is tested. 

## Requirements

Due to the requirements of the *signatory* package, this project uses Python 3.7. We refer you to their project [website](https://pypi.org/project/signatory/) for further information, especially regarding the installation.

