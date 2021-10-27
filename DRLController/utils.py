from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

#Funktion zum schreiben einer gif Datei
def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

#Funktion zum plotten der Rewards and mean Rewards
def plotRewards(Rewards, MeanRewards, fac):
    fig, ax = plt.subplots()

    ax.plot(Rewards, "b-", label="Rewards")
    ax.plot(fac*(np.arange(len(MeanRewards))), MeanRewards, "ko-", label="Mean Rewards")
    ax.set_ylabel('Reward')
    ax.set_xlabel('Episoden')
    ax.legend()
    plt.savefig("Rewards.png")

#Funktionen zum plotten der PID Regler Ergebnisse (P-, I- und D-Term und Gesamtergebnis)
def plotPIDController(pTerm, iTerm, dTerm, PID):
    fig, axs = plt.subplots(2, 2)

    axs[0, 0].plot(pTerm, "b-")
    axs[0, 0].set_ylabel('P-Term')

    axs[0, 1].plot(iTerm, "r-")
    axs[0, 1].set_ylabel('I-Term')

    axs[1, 0].plot(dTerm, "k-")
    axs[1, 0].set_ylabel('D-Term')

    axs[1, 1].plot(PID, "g-")
    axs[1, 1].set_ylabel('PID')

    plt.savefig("PIDResults.png")       