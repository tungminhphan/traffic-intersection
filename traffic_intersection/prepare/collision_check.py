# checks for pedestrian collision
        if (math.sqrt(((pedestrian_1.state[0] - pedestrian_2.state[0]) ** 2) + ((pedestrian_1.state[1] -  pedestrian_2.state[1]) ** 2)) < 76):
            print("There's a collision")
