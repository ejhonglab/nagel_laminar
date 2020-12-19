cmake_minimum_required(VERSION 2.8.3)
project(nagel_laminar)

find_package(catkin REQUIRED)

# Slightly different syntax to what I had used in other catkin projects, but
# this matches the latest wiki example.
install(DIRECTORY
  launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
