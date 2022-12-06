#Ref: https://github.com/ZohebAbai/mobile_sensing_robotics
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from math import sin, cos, atan2, sqrt
import math

def plot_state(mu, S, M):
    # initialize figure
    ax = plt.gca()
    ax.set_xlim([np.min(M[:, 0]) - 2, np.max(M[:, 0]) + 2])
    ax.set_xlim([np.min(M[:, 1]) - 2, np.max(M[:, 1]) + 2])
    plt.plot(M[:, 0], M[:, 1], '^r')

    # visualize result
    plt.plot(mu[0], mu[1], '.b')
    plot_2dcov(mu, S)

def plot_state_2(mu, M):
    # initialize figure
    ax = plt.gca()
    ax.set_xlim([np.min(M[:, 0]) - 2, np.max(M[:, 0]) + 2])
    ax.set_xlim([np.min(M[:, 1]) - 2, np.max(M[:, 1]) + 2])
    plt.plot(M[:, 0], M[:, 1], '^r')

    # visualize result
    plt.plot(mu[0], mu[1], '.b')
def plot_2dcov(mu, cov):
    # covariance only in x,y
    d, v = np.linalg.eig(cov[:-1, :-1])

    # ellipse orientation
    a = np.sqrt(d[0])
    b = np.sqrt(d[1])

    # compute ellipse orientation
    if v[0, 0] == 0:
        theta = np.pi / 2
    else:
        theta = np.arctan2(v[0, 1], v[0, 0])

    # create an ellipse
    ellipse = Ellipse((mu[0], mu[1]),
                      width=a * 2,
                      height=b * 2,
                      angle=np.rad2deg(theta),
                      edgecolor='blue',
                      alpha=0.3)

    ax = plt.gca()

    return ax.add_patch(ellipse)

def plot_2d(mu):

    # create an ellipse
    ellipse = Ellipse((mu[0], mu[1]),
                      width=5 * 2,
                      height=5 * 2,
                      angle=np.rad2deg(1),
                      edgecolor='blue',
                      alpha=0.3)

    ax = plt.gca()

    return ax.add_patch(ellipse)

def wrapToPi(theta): #Warp theta so that theta is always between -pi to pi
    warp_theta=theta
    if warp_theta < -np.pi:
        warp_theta = warp_theta + 2 * np.pi
    if warp_theta > np.pi:
        warp_theta = warp_theta - 2 * np.pi
    return warp_theta

def inv_motion_model(u_t): #Decompose translation into rotation 1， rotation 2 and translation
    translation = sqrt((u_t[1][0] - u_t[0][0]) ** 2 + (u_t[1][1] - u_t[0][1]) ** 2)
    rotation_1 = wrapToPi(atan2((u_t[1][1] - u_t[0][1]), (u_t[1][0] - u_t[0][0])) - u_t[0][2])
    rotation_2 = wrapToPi(u_t[1][2] - u_t[0][2] - rotation_1)

    return rotation_1, translation, rotation_2


def ekf_predict(mu, sigma, u_t, R): #Predict mu and sigma through u_t and R
    theta = mu[2][0]

    rotation_1, translation, rotation_2 = inv_motion_model(u_t)

    G_t = np.array([[1, 0, -translation * sin(theta + rotation_1)],
                    [0, 1, translation * cos(theta + rotation_1)],
                    [0, 0, 1]])

    V_t = np.array([[-translation * sin(theta + rotation_1), cos(theta + rotation_1), 0],
                    [translation * cos(theta + rotation_1), sin(theta + rotation_1), 0],
                    [1, 0, 1]])
    mu_bar = mu + np.array([[translation * cos(theta + rotation_1)],
                            [translation * sin(theta + rotation_1)],
                            [rotation_1 + rotation_2]])

    sigma_bar = np.matmul(G_t ,np.matmul (sigma , G_t.T)) + np.matmul(V_t ,np.matmul(R , V_t.T))

    return mu_bar, sigma_bar


def ekf_correct(mu_bar, sigma_bar, z, Q, M):
    for i in range(z.shape[1]):
        j = int(z[2, i])
        lx = M[j, 0]
        ly = M[j, 1]

        q = (lx - mu_bar[0, 0]) *(lx - mu_bar[0, 0]) + (ly - mu_bar[1, 0]) *(ly - mu_bar[1, 0])
        dist_zi = sqrt(q)

        # wrap to pi as the angle must be between -pi to pi everywhere we deal with the angle

        z_i = np.array([[dist_zi],
                          [wrapToPi(atan2(ly - mu_bar[1, 0], lx - mu_bar[0, 0]) - mu_bar[2, 0])]])

        H_t = np.array(
            [[-(lx - mu_bar[0, 0]) / dist_zi, -(ly - mu_bar[1, 0]) / dist_zi, 0],
             [(ly - mu_bar[1, 0]) / q, -(lx - mu_bar[0, 0]) / q, -1]])

        S_t = np.matmul(H_t ,np.matmul( sigma_bar , H_t.T)) + Q

        K_t = np.matmul(sigma_bar ,np.matmul( H_t.T , np.linalg.inv(S_t))) #compute the Kalman Gain and the most time-consuming step: inversion!

        mu_bar = mu_bar + (np.matmul(K_t , (z[:2, i].reshape(2, 1) - z_i))) #The correct step
        mu_bar[2, 0] = wrapToPi(mu_bar[2, 0])
        sigma_bar = np.matmul((np.identity(3) - np.matmul(K_t , H_t)) , sigma_bar) #Reduce the uncertainty

    return mu_bar, sigma_bar

#Estimate the robot's trajectory from only sensor measurement
def ekf_predict_z( z, Q, M):
    new_mu=np.zeros((2,1))
    for i in range(z.shape[1]):
        j = int(z[2, i])
        lx = M[j, 0]
        ly = M[j, 1]
        new_mu=new_mu+np.array([lx-z[0,i],ly-z[1,i]]).reshape((2,1))
        sigma_bar=Q
    mu=new_mu/z.shape[1]
    mu=np.append(mu,0)
    return mu