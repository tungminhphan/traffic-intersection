# Collision Detection Cars and Pedestrians
# Anhminh Nguyen
# July 10, 2018


# checks for pedestrian collision
def collision_check():
        if (math.sqrt(((pedestrian_1.state[0] - pedestrian_2.state[0]) ** 2) + ((pedestrian_1.state[1] -  pedestrian_2.state[1]) ** 2)) < 76):
            print("There's a collision")
