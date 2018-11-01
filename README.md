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
```

## Built With

* [Montblanc](https://github.com/ska-sa/montblanc/) - Interferomtry simulation using RIME
* [PyEphem](https://rhodesmill.org/pyephem/) - Astronomical position calculations
* [uvgen](https://github.com/SpheMakh/uvgen) - U,V coordinate simulation

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* **Chris Finlay** - *Initial work* - [Github](https://github.com/chrisfinlay)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Simon Perkins (For all the help with understanding Montblanc, his baby)
* Michelle Lochner 
* Bruce Bassett


