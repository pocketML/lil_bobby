import matplotlib.pyplot as plt
import matplotlib.patches as patches
font = {'size': 16}
plt.rc('font', **font)

PLOT_FAST = True

def parabola(x, a, b, c):
    return a * x ** 2 + b * x + c

def plot_steps(steps_x, steps_y, ax):
    for index, (x_s, y_s) in enumerate(zip(steps_x, steps_y)):
        if index > 0:
            prev_x = steps_x[index-1]
            prev_y = steps_y[index-1]
            rad="-.5" if prev_x < x_s else ".5"
            arrow = patches.FancyArrowPatch(
                (prev_x, prev_y), (x_s, y_s),
                connectionstyle=f"arc3,rad={rad}", **kw
            )
            ax.add_patch(arrow)

fig, ax = plt.subplots(1, 1)

p_a = 2
p_b = 1
p_c = 0

points = 200
mid = 0
x = [p for p in range(mid - points // 2, mid + points // 2)]
y = [parabola(p, p_a, p_b, p_c) for p in x]

ax.plot(x, y)

arrow_style = "Simple, tail_width=0.5, head_width=4, head_length=8"
kw = dict(arrowstyle=arrow_style, color="red")

if PLOT_FAST:
    ax.set_title("Learning Rate Too High")
    plot_name = "grad_desc_lr_high"
    x_steps_fast = [-58, -47, -35, -15, 15, -10, 10]
    y_steps_fast = [parabola(p, p_a, p_b, p_c) for p in x_steps_fast]
    plot_steps(x_steps_fast, y_steps_fast, ax)
else:
    ax.set_title("Learning Rate Too Low")
    plot_name = "grad_desc_lr_low"
    x_steps_slow = [-58, -50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0]
    y_steps_slow = [parabola(p, p_a, p_b, p_c) for p in x_steps_slow]
    plot_steps(x_steps_slow, y_steps_slow, ax)

ax.set_ylabel("Error")

ax.set_xlim(-80, 80)
ax.set_ylim(-100, 7000)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect(0.015)
ax.set_xlabel("Parameters")
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,
    top=False,
    left=False,
    right=False
)

plt.tight_layout()
fig.set_size_inches(6, 4.5, forward=True)
fig.savefig(f"misc/{plot_name}.pdf", bbox_inches="tight")
plt.show()
