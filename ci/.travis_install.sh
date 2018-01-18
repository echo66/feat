
#installing gcc and g++ version 5 and setting them as default
# sudo add-apt-repository --yes ppa:ubuntu-toolchain-r/test -y
# sudo apt-get update
# sudo apt-get install gcc-5 g++-5
# sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 60 --slave /usr/bin/g++ g++ /usr/bin/g++-5
# sudo apt-get install -qq g++-5
# sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 1
#  #- sudo rm /usr/bin/g++
# sudo ln -s /usr/bin/g++-5 /usr/bin/g++  
  
#installing cmake
  #- sudo apt-get install cmake
  #- g++-5 --version
  #- cmake --version
  #- sudo apt-get install build-essential
  #- wget http://www.cmake.org/files/v3.2/cmake-3.2.2.tar.gz
  #- tar xf cmake-3.2.2.tar.gz
  #- cd cmake-3.2.2
  #- ./configure
  #- make
  #- sudo apt-get install checkinstall
  #- sudo checkinstall -y
  #- dpkg-query -l '*cmake*'
  #- sudo apt-get install cmake
    # sudo apt-get install cmake-data
    
#______________________________________________    
 #sudo apt-get install software-properties-common
 #sudo add-apt-repository ppa:george-edison55/cmake-3.x -y
 #sudo apt-get update
 #sudo apt-get install --only-upgrade cmake
 #dpkg-query -l '*cmake*'
#installing eigen 3
 #sudo wget "http://bitbucket.org/eigen/eigen/get/3.3.4.tar.gz" -O- | sudo tar xvz -C /usr/include/
 #sudo wget "http://bitbucket.org/eigen/eigen/get/3.3.4.tar.gz"
 #sudo tar xvzf 3.3.4.tar.gz
 #cd 3.3.4
 #ls
 #pwd
 #cd eigen*
 #pwd
 #ls
 #sudo mkdir build_dir
 #pwd
 #cd build_dir
 #pwd
 #sudo cmake CMakeLists.txt
 #sudo make install
 #dpkg-query -l '*eigen*'
#_______________________________________________
  
#installing shogun library
  #- sudo add-apt-repository ppa:shogun-toolbox/stable -y
  #- sudo apt-get update -y
  #- sudo apt-get install libshogun-dev
  #- sudo apt-get install libshogun18 -y
  #- sudo apt-get install libshogun16 -y
  #- sudo apt-get install python2.7-shogun
  #- sudo dpkg-query -l '*shogun*'
# sudo apt-get update

#______________________________________________________
 sudo apt-get install -qq software-properties-common lsb-release
 sudo add-apt-repository "deb http://archive.ubuntu.com/ubuntu $(lsb_release -sc) multiverse"
 sudo add-apt-repository ppa:shogun-toolbox/nightly -y
 sudo apt-get update -y
 sudo apt-get install -qq --force-yes --no-install-recommends libshogun18
 sudo apt-get install -qq --force-yes --no-install-recommends libshogun-dev
 sudo apt-get install -qq --force-yes --no-install-recommends python-shogun
 sudo dpkg-query -l '*shogun*'
  
#building and installing google tests
 sudo apt-get install cmake
 sudo apt-get install libgtest-dev
 
 cmake --version
 pwd
 cd /usr/src/gtest
 sudo cmake CMakeLists.txt
 sudo make
 sudo cp *.a /usr/lib
 cd /home/travis/build/lacava/fewtwo
 pwd
 
 sudo apt-get install libeigen3-dev
 
 #sudo wget "http://bitbucket.org/eigen/eigen/get/3.3.4.tar.gz"
 #sudo tar xvzf 3.3.4.tar.gz
 #cd 3.3.4
 #ls
 #pwd
 #cd eigen*
 #sudo mkdir build
 #pwd
 #ls
 #cd build
 #pwd
 #sudo cmake ..
 #ls
 #cat Makefile
 #sudo make install
#_________________________________________________________

