"""

# Currently not used, but could be fun for experimentation!
                com = obs['com']                      # [x, y, z]
                foot_contacts = obs['foot_contacts']  # [foot1, foot2, ... , foot6]
                joint_sensors = obs['joint_sensors']  # Rotational Motor Angles

                ----------------------------------------------------------------------------

                #Lidar code
                lidar_values_original = obs['lidar']
                Sanitizing values to avoid inf when robot flips over
                lidar_values = [v if math.isfinite(v) else 999 for v in lidar_values_original]

                [...]

                # Acceptable height + stability at height
                    h_base = 3 # empirically defined as reasonable height
                    diff = abs(lidar_values[1] - h_base)

                ----------------------------------------------------------------------------

                # For future implementations:
                # The more stable, the higher the reward
                # In order to avoid explosive increase,
                # we consider 20% of total steps being stable
                # as multiplier for reward
                # (1 + (0.005 * self.stable_counter)) if self.is_tilted==False


                ----------------------------------------------------------------------------

                 # For every foot that's not touching the ground, we take points
                for v in foot_contacts:
                if v == 0: reward -= 0.5
                elif v == 1: reward += 1
                else: print("NON-READABLE FOOT SENSOR VALUE! ", v)

                ----------------------------------------------------------------------------

             Following the mantis tutorial, after reading the .wbt
              file values for the hinge position:
               - if the "elbow" hinges were to be perfectly bent/balanced,
               its angle would be [ ~ -2.4121293759260714 rad -> ~ -138.2 deg ]

             Therefore, the lower this negative number is, the tighter the
              robot closes its arm.

             In order to, again, respect a threshold as it was done in
              the height check.

             However, the correct sensor placement is being blocked by internal
             Webots processes. This makes it impossible to measure.

                ----------------------------------------------------------------------------

            # Acceptable arm position (hinge safety and correct execution of task)
            aC, aF, aT = 0.25, 0.20, 0.05  # perfect value amplitudes
            dC, dF, dT = 0.60, 0.80, -2.40  # offsets (centers)
            minC, maxC = dC - aC, dC + aC
            minF, maxF = dF - aF, dF + aF
            minT, maxT = dT - aT, dT + aT

            # Shoulders are not as important as elbow bend, so their reward is smaller
            for i in range(18):
                if i < 6:
                    if minC <= joint_sensors[i] <= maxC:
                        # print("Considers perfect interval for base!")
                        reward += 0.2
                    else: reward -= 1

                elif i < 12:
                    if minF <= joint_sensors[i] <= maxF:
                        reward += 0.2
                    else: reward -= 1

                elif i < 18:
                    if minT <= joint_sensors[i] <= maxT:
                        print("Considers perfect interval for elbow!")
                        reward += 0.2
                    else: reward -= 1

"""