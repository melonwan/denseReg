# for matplotlib wrapper to tf summary
import tensorflow as tf
import tfplot, matplotlib
import matplotlib.lines as lines

FLAGS = tf.app.flags.FLAGS

def figure_heatmap(hm):
    fig = matplotlib.figure.Figure()
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(hm, cmap=matplotlib.cm.jet)
    fig.colorbar(im)
    return fig

def figure_joint(dm, uvd_pt):
    fig = matplotlib.figure.Figure()
    ax = fig.add_subplot(1,1,1)
    ax.imshow(dm, cmap=matplotlib.cm.Greys)

    if FLAGS.dataset == 'bighand':
        ax.scatter(uvd_pt[0,0], uvd_pt[0,1], s=200, c='w')
        ax.scatter(uvd_pt[1:6,0], uvd_pt[1:6,1], s=100, c='w')
        ax.scatter(uvd_pt[6:9,0], uvd_pt[6:9,1], s=60, c='c')
        ax.scatter(uvd_pt[9:12,0], uvd_pt[9:12,1], s=60, c='m')
        ax.scatter(uvd_pt[12:15,0], uvd_pt[12:15,1], s=60, c='y')
        ax.scatter(uvd_pt[15:18,0], uvd_pt[15:18,1], s=60, c='g')
        ax.scatter(uvd_pt[18:,0], uvd_pt[18:,1], s=60, c='r')
    elif FLAGS.dataset == 'nyu':
        ax.scatter(uvd_pt[10:,0], uvd_pt[10:,1], s=200, c='w')
        ax.scatter(uvd_pt[0,0], uvd_pt[0,1], s=60, c='c')
        ax.scatter(uvd_pt[1,0], uvd_pt[1,1], s=90, c='c')
        ax.scatter(uvd_pt[2,0], uvd_pt[2,1], s=60, c='m')
        ax.scatter(uvd_pt[3,0], uvd_pt[3,1], s=90, c='m')
        ax.scatter(uvd_pt[4,0], uvd_pt[4,1], s=60, c='y')
        ax.scatter(uvd_pt[5,0], uvd_pt[5,1], s=90, c='y')
        ax.scatter(uvd_pt[6,0], uvd_pt[6,1], s=60, c='g')
        ax.scatter(uvd_pt[7,0], uvd_pt[7,1], s=90, c='g')
        ax.scatter(uvd_pt[8,0], uvd_pt[8,1], s=60, c='r')
        ax.scatter(uvd_pt[9,0], uvd_pt[9,1], s=90, c='r')
    elif FLAGS.dataset == 'msra':
        fig_color = ['c', 'm', 'y', 'g', 'r']
        ax.scatter(uvd_pt[0:,0], uvd_pt[0:,1], s=200, c='w')
        for f in range(5):
            ax.scatter(uvd_pt[f*4+1,0], uvd_pt[f*4+1,1], s=90, c=fig_color[f])
            ax.scatter(uvd_pt[f*4+2,0], uvd_pt[f*4+2,1], s=80, c=fig_color[f])
            ax.scatter(uvd_pt[f*4+3,0], uvd_pt[f*4+3,1], s=70, c=fig_color[f])
            ax.scatter(uvd_pt[f*4+4,0], uvd_pt[f*4+4,1], s=60, c=fig_color[f])

    elif FLAGS.dataset == 'icvl':
        fig_color = ['c', 'm', 'y', 'g', 'r']
        ax.scatter(uvd_pt[0:,0], uvd_pt[0:,1], s=200, c='w')
        for f in range(5):
            ax.scatter(uvd_pt[f*3+1,0], uvd_pt[f*3+1,1], s=90, c=fig_color[f])
            ax.scatter(uvd_pt[f*3+2,0], uvd_pt[f*3+2,1], s=80, c=fig_color[f])
            ax.scatter(uvd_pt[f*3+3,0], uvd_pt[f*3+3,1], s=60, c=fig_color[f])
    return fig

def figure_joint_skeleton(dm, uvd_pt):
    fig = matplotlib.figure.Figure()
    ax = fig.add_subplot(1,1,1)
    ax.imshow(dm, cmap=matplotlib.cm.Greys)

    if FLAGS.dataset == 'bighand':
        ax.scatter(uvd_pt[0,0], uvd_pt[0,1], s=200, c='w')
        ax.scatter(uvd_pt[1:6,0], uvd_pt[1:6,1], s=100, c='w')
        ax.scatter(uvd_pt[6:9,0], uvd_pt[6:9,1], s=60, c='c')
        ax.scatter(uvd_pt[9:12,0], uvd_pt[9:12,1], s=60, c='m')
        ax.scatter(uvd_pt[12:15,0], uvd_pt[12:15,1], s=60, c='y')
        ax.scatter(uvd_pt[15:18,0], uvd_pt[15:18,1], s=60, c='g')
        ax.scatter(uvd_pt[18:,0], uvd_pt[18:,1], s=60, c='r')
    elif FLAGS.dataset == 'nyu':
        fig_color = ['c', 'm', 'y', 'g', 'r']
        for f in range(5):
            ax.plot([uvd_pt[f*2,0], uvd_pt[f*2+1,0]], 
                    [uvd_pt[f*2,1], uvd_pt[f*2+1,1]], color=fig_color[f], linewidth=3)
            ax.scatter(uvd_pt[f*2,0],uvd_pt[f*2,1],s=60,c=fig_color[f])
            ax.scatter(uvd_pt[f*2+1,0],uvd_pt[f*2+1,1],s=60,c=fig_color[f])
            if f<4:
                ax.plot([uvd_pt[13,0], uvd_pt[f*2+1,0]], 
                        [uvd_pt[13,1], uvd_pt[f*2+1,1]], color=fig_color[f], linewidth=3)
        ax.plot([uvd_pt[9,0], uvd_pt[10,0]], 
                [uvd_pt[9,1], uvd_pt[10,1]], color='r', linewidth=3)

        ax.scatter(uvd_pt[13,0], uvd_pt[13,1], s=200, c='w')
        ax.scatter(uvd_pt[11,0], uvd_pt[11,1], s=100, c='b')
        ax.scatter(uvd_pt[12,0], uvd_pt[12,1], s=100, c='b')

        ax.plot([uvd_pt[13,0], uvd_pt[11,0]], 
                [uvd_pt[13,1], uvd_pt[11,1]], color='b', linewidth=3)
        ax.plot([uvd_pt[13,0], uvd_pt[12,0]], 
                [uvd_pt[13,1], uvd_pt[12,1]], color='b', linewidth=3)
        ax.plot([uvd_pt[13,0], uvd_pt[10,0]], 
                [uvd_pt[13,1], uvd_pt[10,1]], color='r', linewidth=3)

    elif FLAGS.dataset == 'msra':
        fig_color = ['c', 'm', 'y', 'g', 'r']
        ax.scatter(uvd_pt[0:,0], uvd_pt[0:,1], s=200, c='w')
        for f in range(5):
            ax.scatter(uvd_pt[f*4+1,0], uvd_pt[f*4+1,1], s=90, c=fig_color[f])
            ax.scatter(uvd_pt[f*4+2,0], uvd_pt[f*4+2,1], s=80, c=fig_color[f])
            ax.scatter(uvd_pt[f*4+3,0], uvd_pt[f*4+3,1], s=70, c=fig_color[f])
            ax.scatter(uvd_pt[f*4+4,0], uvd_pt[f*4+4,1], s=60, c=fig_color[f])
            ax.plot([uvd_pt[f*4+1,0], uvd_pt[f*4+2,0]],
                    [uvd_pt[f*4+1,1], uvd_pt[f*4+2,1]], color=fig_color[f], linewidth=3)
            ax.plot([uvd_pt[f*4+2,0], uvd_pt[f*4+3,0]],
                    [uvd_pt[f*4+2,1], uvd_pt[f*4+3,1]], color=fig_color[f], linewidth=3)
            ax.plot([uvd_pt[f*4+3,0], uvd_pt[f*4+4,0]],
                    [uvd_pt[f*4+3,1], uvd_pt[f*4+4,1]], color=fig_color[f], linewidth=3)
    elif FLAGS.dataset == 'icvl':
        fig_color = ['c', 'm', 'y', 'g', 'r']
        ax.scatter(uvd_pt[0:,0], uvd_pt[0:,1], s=200, c='w')
        for f in range(5):
            ax.scatter(uvd_pt[f*3+1,0], uvd_pt[f*3+1,1], s=90, c=fig_color[f])
            ax.scatter(uvd_pt[f*3+2,0], uvd_pt[f*3+2,1], s=80, c=fig_color[f])
            ax.scatter(uvd_pt[f*3+3,0], uvd_pt[f*3+3,1], s=60, c=fig_color[f])
            ax.plot([uvd_pt[f*3+1,0], uvd_pt[f*3+2,0]],
                    [uvd_pt[f*3+1,1], uvd_pt[f*3+2,1]], color=fig_color[f], linewidth=3)
            ax.plot([uvd_pt[f*3+2,0], uvd_pt[f*3+3,0]],
                    [uvd_pt[f*3+2,1], uvd_pt[f*3+3,1]], color=fig_color[f], linewidth=3)

    return fig

def figure_smp_pts(dm, pts1, pts2):
    fig = matplotlib.figure.Figure()
    ax = fig.add_subplot(1,1,1)
    ax.imshow(dm, cmap=matplotlib.cm.jet)

    for pt1, pt2 in zip(pts1, pts2):
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])
        ax.scatter(pt1[0], pt1[1], s=60, c='w')
        ax.scatter(pt2[0], pt2[1], s=60, c='m')
    return fig

tf_heatmap_wrapper = tfplot.wrap(figure_heatmap, batch=True, name='hm_summary')
tf_jointplot_wrapper = tfplot.wrap(figure_joint_skeleton, batch=True, name='pt_summary')
tf_smppt_wrapper = tfplot.wrap(figure_smp_pts, batch=True, name='smppt_summary')

