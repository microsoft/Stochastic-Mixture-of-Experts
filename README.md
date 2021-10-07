# THOR: Transformer with Stochastic Experts

This PyTorch package implements Taming Sparsely Activated Transformer with Stochastic Experts.

## Installation
* The most convenient way to run the code is to use this docker image: `tartarusz/adv-train:azure-pytorch-apex-v1.7.0`. 
  The image supports running on Microsoft Azure.
* Our implementation is based on [Fairseq](https://github.com/pytorch/fairseq).

## Instructions
* Download [Fairseq](https://github.com/pytorch/fairseq) (v1.0.0+) to the current directory.
* Run `pip install -e .` to install the package locally.
* To run a sample translation task on IWSLT'14 De-En, 
  first follow the instructions [here](https://github.com/pytorch/fairseq/blob/main/examples/translation/README.md)
  to download and tokenize the data, then use `bash preprocess.sh` to pre-process the tokenized data.
* Run `bash run.sh` to train a THOR model.

## Notes

### Contact Information

For personal communication related to this package, please contact Simiao Zuo (`simiaozuo@gatech.edu`), Xiaodong Liu (`xiaodl@microsoft.com`), or Jian Jiao (`jian.jiao@microsoft.com`).

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
