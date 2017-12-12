/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <iostream>
#include <sstream>

#include "particle_filter.h"

void ParticleFilter::init(const double x, const double y, const double theta, const double std[])
{
	num_particles = 50;

	std::default_random_engine gen;

	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);

	particles.resize(num_particles);
	weights.resize(num_particles);

	for (int i = 0; i < num_particles; i++)
	{
		Particle &particle = particles[i];
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1.0;

		weights[i] = 1.0;
	}

	is_initialized = true;
}

void ParticleFilter::prediction(const double delta_t, const double std_pos[], const double velocity, double yaw_rate)
{
	std::default_random_engine gen_pred;

	std::normal_distribution<double> dist_x(0, std_pos[0]);
	std::normal_distribution<double> dist_y(0, std_pos[1]);
	std::normal_distribution<double> dist_theta(0, std_pos[2]);

	if (fabs(yaw_rate) < 0.0001)
	{
		yaw_rate = 0.0001;
	}

	for (auto &particle : particles)
	{
		particle.x += dist_x(gen_pred) + velocity / yaw_rate * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
		particle.y += dist_y(gen_pred) + velocity / yaw_rate * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
		particle.theta += dist_theta(gen_pred) + yaw_rate * delta_t;
	}
}

void ParticleFilter::dataAssociation(const std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations)
{
	for (auto &observation : observations)
	{
		double nearest = std::numeric_limits<double>::max();
		for (int i = 0; i < predicted.size(); i++)
		{
			const LandmarkObs &pred = predicted[i];
			double dist = sqrt(pow(observation.x - pred.x, 2) + pow(observation.y - pred.y, 2));
			if (dist < nearest)
			{
				nearest = dist;
				observation.id = i;
			}
		}
	}
}

void ParticleFilter::updateWeights(const double sensor_range, const double std_landmark[],
								   const std::vector<LandmarkObs> &observations, const Map &map_landmarks)
{
	for (int i = 0; i < num_particles; i++)
	{
		Particle &particle = particles[i];
		const double p_x = particle.x;
		const double p_y = particle.y;
		const double p_theta = particle.theta;

		std::vector<LandmarkObs> predicted_landmarks;
		for (auto &landmark : map_landmarks.landmark_list)
		{
			if (sensor_range > dist(p_x, p_y, landmark.x_f, landmark.y_f))
			{
				const LandmarkObs lm{landmark.id_i, landmark.x_f, landmark.y_f};
				predicted_landmarks.push_back(lm);
			}
		}

		std::vector<LandmarkObs> new_observations;
		for (auto &observation : observations)
		{
			LandmarkObs new_observation;
			new_observation.x = p_x + observation.x * cos(p_theta) - observation.y * sin(p_theta);
			new_observation.y = p_y + observation.x * sin(p_theta) + observation.y * cos(p_theta);
			new_observations.push_back(new_observation);
		}

		dataAssociation(predicted_landmarks, new_observations);

		particle.weight = 1.0;
		for (auto &observation : new_observations)
		{
			const LandmarkObs &landmark = predicted_landmarks[observation.id];

			const double diff_x_2 = pow(observation.x - landmark.x, 2);
			const double diff_y_2 = pow(observation.y - landmark.y, 2);
			const double sig_x = std_landmark[0];
			const double sig_y = std_landmark[1];
			const double gauss_norm = (1 / (2 * M_PI * sig_x * sig_y));
			const double exponent = diff_x_2 / (2 * sig_x * sig_x) + diff_y_2 / (2 * sig_y * sig_y);

			particle.weight *= gauss_norm * exp(-exponent);
		}

		weights[i] = particle.weight;
	}
}

void ParticleFilter::resample()
{
	std::default_random_engine gen;
	std::discrete_distribution<> disc_dist(weights.begin(), weights.end());

	std::vector<Particle> resampled_particles;
	for (const auto &p : particles)
	{
		resampled_particles.push_back(particles[disc_dist(gen)]);
	}
	particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle &particle, const std::vector<int> &associations,
										 const std::vector<double> &sense_x, const std::vector<double> &sense_y)
{
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;
}

std::string ParticleFilter::getAssociations(Particle best)
{
	std::vector<int> v = best.associations;
	std::stringstream ss;
	copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
	std::string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}

std::string ParticleFilter::getSenseX(Particle best)
{
	std::vector<double> v = best.sense_x;
	std::stringstream ss;
	copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
	std::string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}

std::string ParticleFilter::getSenseY(Particle best)
{
	std::vector<double> v = best.sense_y;
	std::stringstream ss;
	copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
	std::string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
