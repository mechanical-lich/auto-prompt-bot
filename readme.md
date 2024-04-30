# "Pi Artist"

# Installation
 - It is recommended to start with a headless version of Rasberry Pi OS.   Other distros may work, but shoot for anything that can run headless to save resources.
 - Install python dependencies `pip install -r requirements.txt`
 - Configure Swap
    ```
    sudo dphys-swapfile swapoff
    sudo nano /etc/dphys-swapfile
    ```
    - Update variable `CONF_SWAPSIZE` to 8192
    - Uncomment and increase `CONF_MAXSWAP` to 8192
    ```
    sudo dphys-swapfile setup
    sudo dphys-swapfile swapon
    ```

# Configuration


# Running
` python main.py`