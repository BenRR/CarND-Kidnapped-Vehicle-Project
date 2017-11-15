/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  // Number of particles to draw
  num_particles = 65;

  default_random_engine gen;

  normal_distribution<double> noise_x(0, std[0]);
  normal_distribution<double> noise_y(0, std[1]);
  normal_distribution<double> noise_t(0, std[2]);

  for (int i = 0; i < num_particles; ++i) {
    Particle p = {
      i,
      x + noise_x(gen),
      y + noise_y(gen),
      theta + noise_t(gen),
      1.0
    };
    weights.push_back(1.0);
    particles.push_back(p);

  }

  // Flag, if filter is initialized
  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/
  default_random_engine gen;
  normal_distribution<double> noise_x(0, std_pos[0]);
  normal_distribution<double> noise_y(0, std_pos[1]);
  normal_distribution<double> noise_t(0, std_pos[2]);

  for (int i = 0; i < num_particles; i++) {

    double t = particles[i].theta;

    if (fabs(yaw_rate) < 0.0001) {
      // go straight
      particles[i].x += delta_t * velocity * cos(t) + noise_x(gen);
      particles[i].y += delta_t * velocity * sin(t) + noise_y(gen);
    } else {

      double nt = t + yaw_rate * delta_t;

      particles[i].x += velocity / yaw_rate * (sin(nt) - sin(t)) + noise_x(gen);
      particles[i].y += velocity / yaw_rate * (cos(t) - cos(nt)) + noise_y(gen);
      particles[i].theta = nt + noise_t(gen);
    }
  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.
  for (int i = 0; i < observations.size(); ++i) {
    int idx = -1;
    double current_error = -1;
    for (int j = 0; j < predicted.size(); ++j) {
      double error = pow(predicted[j].x - observations[i].x, 2) + pow(predicted[j].y - observations[i].y, 2);
      if (j == 0 || error < current_error) {
        idx = j;
        current_error = error;
      }
    }

    observations[i].id = idx;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html



  for (int i = 0; i < particles.size(); ++i) {

    // find landmarks in reach use euclidean distance
    vector<LandmarkObs> landmarks_in_reach;
    for (int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
      double distance = sqrt(pow(map_landmarks.landmark_list[j].x_f - particles[i].x, 2) +
                             pow(map_landmarks.landmark_list[j].y_f - particles[i].y, 2));
      if (distance < sensor_range) {
        landmarks_in_reach.push_back(LandmarkObs{map_landmarks.landmark_list[j].id_i,
                                                 map_landmarks.landmark_list[j].x_f,
                                                 map_landmarks.landmark_list[j].y_f});
      }
    }

    //transform observations to map coordinates
    vector<LandmarkObs> transformed_observations;
    for (int k = 0; k < observations.size(); ++k) {
      double n_x =
        particles[i].x + observations[k].x * cos(particles[i].theta) - sin(particles[i].theta) * observations[k].y;
      double n_y =
        particles[i].y + observations[k].x * sin(particles[i].theta) + cos(particles[i].theta) * observations[k].y;
      transformed_observations.push_back(LandmarkObs{observations[k].id, n_x, n_y});
    }

    // link observations with landmarks
    dataAssociation(landmarks_in_reach, transformed_observations);


    // update weights
    double weight = 1.0;
    double std_x = std_landmark[0];
    double std_y = std_landmark[1];

    for (int l = 0; l < transformed_observations.size(); ++l) {
      double observ_x = transformed_observations[l].x;
      double observ_y = transformed_observations[l].y;

      double l_idx = transformed_observations[l].id;
      double landmark_x = landmarks_in_reach[l_idx].x;
      double landmark_y = landmarks_in_reach[l_idx].y;

      weight *= (1 / (2 * M_PI * std_x * std_y)) *
                exp(-(pow(landmark_x - observ_x, 2) / (2 * pow(std_x, 2)) +
                      (pow(landmark_y - observ_y, 2) / (2 * pow(std_y, 2)))));
    }

    particles[i].weight = weight;
    weights[i] = weight;
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  vector<Particle> resampled;
  default_random_engine gen;
  discrete_distribution<int> dist(weights.begin(), weights.end());

  for (int i = 0; i < particles.size(); ++i) {
    int idx = dist(gen);
    resampled.push_back(particles[idx]);
  }
  particles = resampled;

  for (int j = 0; j < weights.size(); ++j) {
    weights[j] = particles[j].weight;
  }

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
