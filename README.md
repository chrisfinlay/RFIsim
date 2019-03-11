# RFIsim

This package is an RFI simulation tool for the MeerKAT radio telescope. It calculates real world satellite movement using two-line element sets (TLEs) and simulates a frequency spectrum and time dependence.

The output are raw visibilities that are saved into a HDF5 file along with all the input data used for the simulation.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You will only need [Docker](https://docs.docker.com/install/) for this.

If you have an Nvidia GPU capable of running TensorFlow code then make sure you have [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) installed.

##### NB GPU version currently has bugs

### Installing

To get the docker environment running to use this tool simply run the following command.

```
nvidia-docker run -it -v dir/on/host:/data chrisjfinlay/montblanc:ddfacet-fixed-gpu
cd /data
git clone https://github.com/chrisfinlay/RFIsim.git
cd /data/RFIsim/utils/astronomical/catalogues
unzip SUMSS_NVSS_Clean.csv.zip
cd /data/RFIsim/utils/telescope/beam_sim
mkdir beams
python create_beam.py
cd /data/RFIsim
python RFIsim.py
```

### Output Data

#### Structure
```
/
|--- input
|       |--- target                  (RA, DEC)
|       |--- astro_sources           (n_astro_srcs, 7)
|       |--- astro_sources_headings  (7)
|       |--- rfi_lm                  (time_steps, n_rfi_srcs, {l,m})
|       |--- rfi_spectra             (n_rfi_srcs, time_steps, freq_chans, n_pols)
|       |--- UVW                     (time_steps*n_bl, {u,v,w})
|       |--- A1                      (n_bl)
|       |--- A2                      (n_bl)
|       |--- bandpass                (time_steps, n_ants, freq_chans, n_pols)
|       |--- frequencies             (freq_chans)
|       |--- auto_pol_gains          (1)
|       |--- cross_pol_gains         (1)
|--- output
        |--- vis_clean               (time_steps, n_bl, freq_chans, n_pols)
        |--- vis_dirty               (time_steps, n_bl, freq_chans, n_pols)
```

To get the DataFrame of astronomical source parameters used in the simulation.
```
with h5py.File('test.h5', 'r') as fp:
    srcs_df = pd.DataFrame(data=fp['input/astro_sources'].value,
                           columns=fp['input/astro_sources_headings'].value)
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
* NVSS
* SUMSS
