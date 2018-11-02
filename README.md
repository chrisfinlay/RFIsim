# RFIsim

This package is an RFI simulation tool for the MeerKAT radio telescope. It calculates real world satellite movement using two-line element sets (TLEs) and simulates a frequency spectrum and time dependence.

The output are raw visibilities in the form of a numpy array that is saved as a binary .npy file.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

You will only need [Docker](https://docs.docker.com/install/) for this.

If you have an Nvidia GPU capable of running TensorFlow code then make sure you have [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) installed.

### Installing

To get the docker environment running to use this tool simply run the following command.

```
docker run -it -v dir/on/host:/data chrisjfinlay/montblanc:ddfacet-py2-new
cd /data
git clone https://github.com/chrisfinlay/RFIsim.git
cd RFIsim
python RFIsim.py
```

## Built With

* [Montblanc](https://github.com/ska-sa/montblanc/) - Interferometry simulation using RIME
* [PyEphem](https://rhodesmill.org/pyephem/) - Astronomical position calculations
* [uvgen](https://github.com/SpheMakh/uvgen) - U,V coordinate simulation

## Authors

* **Chris Finlay** - [LinkedIn](https://www.linkedin.com/in/chris-finlay/)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Simon Perkins (For all the help with understanding Montblanc, his baby)
* Michelle Lochner
* Bruce Bassett
* Nadeem Oozeer
