from matplotlib.path import get_path_collection_extents
from textalloc.non_overlapping_boxes import (
    get_non_overlapping_boxes,
    find_nearest_point_on_box,
)
import numpy as np
import time
from typing import Any, Dict, List, Union
import warnings

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterator, *args, **kwargs):
        return iterator


def allocate(
    ax,
    x: Union[np.ndarray, List[float]],
    y: Union[np.ndarray, List[float]],
    text_list: List[str],
    x_scatter: Union[np.ndarray, List[float]] = None,
    y_scatter: Union[np.ndarray, List[float]] = None,
    x_lines: List[Union[np.ndarray, List[float]]] = None,
    y_lines: List[Union[np.ndarray, List[float]]] = None,
    scatter_sizes: List[Union[np.ndarray, List[float]]] = None,
    scatter_plot: object = None,
    text_scatter_sizes: List[Union[np.ndarray, List[float]]] = None,
    textsize: Union[int, List[int]] = 10,
    margin: float = 0.008,
    min_distance: float = 0.013,
    max_distance: float = 0.2,
    verbose: bool = False,
    draw_lines: bool = True,
    linecolor: Union[str, List[str]] = "r",
    draw_all: bool = True,
    nbr_candidates: int = 200,
    linewidth: float = 1,
    textcolor: Union[str, List[str]] = "k",
    seed: int = 0,
    direction: str = None,
    avoid_label_lines_overlap: bool = False,
    plot_kwargs: Dict[str, Any] = None,
    **kwargs,
):
    """Main function of allocating text-boxes in matplotlib plot

    Args:
        ax (_type_): matplotlib axes used for plotting.
        x (Union[np.ndarray, List[float]]): x-coordinates of texts 1d array/list.
        y (Union[np.ndarray, List[float]]): y-coordinates of texts 1d array/list.
        text_list (List[str]): list of texts.
        x_scatter (Union[np.ndarray, List[float]], optional): x-coordinates of all scattered points in plot 1d array/list. Defaults to None.
        y_scatter (Union[np.ndarray, List[float]], optional): y-coordinates of all scattered points in plot 1d array/list. Defaults to None.
        x_lines (List[Union[np.ndarray, List[float]]], optional): x-coordinates of all lines in plot list of 1d arrays/lists. Defaults to None.
        y_lines (List[Union[np.ndarray, List[float]]], optional): y-coordinates of all lines in plot list of 1d arrays/lists. Defaults to None.
        scatter_sizes (List[Union[np.ndarray, List[float]]], optional): sizes of all scattered objects in plot list of 1d arrays/lists. Defaults to None.
        scatter_plot (__type__, optional): if possible, provide a scatterplot object (scatter_plot=ax.scatter(...)) for more exact placement. Defaults to None.
        text_scatter_sizes (List[Union[np.ndarray, List[float]]], optional): sizes of text scattered objects in plot list of 1d arrays/lists. Defaults to None.
        textsize (Union[int, List[int]], optional): size of text. Defaults to 10.
        margin (float, optional): parameter for margins between objects. Increase for larger margins to points and lines. Defaults to 0.008.
        min_distance (float, optional): parameter for min distance between text and origin. Defaults to 0.015.
        max_distance (float, optional): parameter for max distance between text and origin. Defaults to 0.2.
        verbose (bool, optional): prints progress using tqdm. Defaults to False.
        draw_lines (bool, optional): draws lines from original points to textboxes. Defaults to True.
        linecolor (Union[str, List[str]], optional): color code of the lines between points and text-boxes. Defaults to "r".
        draw_all (bool, optional): Draws all texts after allocating as many as possible despit overlap. Defaults to True.
        nbr_candidates (int, optional): Sets the number of candidates used. Defaults to 200.
        linewidth (float, optional): width of line. Defaults to 1.
        textcolor (Union[str, List[str]], optional): color code of the text. Defaults to "k".
        seed (int, optional): seeds order of text allocations. Defaults to 0.
        direction (str, optional): set preferred location of the boxes (south, north, east, west, northeast, northwest, southeast, southwest). Defaults to None.
        avoid_label_lines_overlap (bool, optional): If True, avoids overlap with lines drawn between text labels and locations. Defaults to False.
        plot_kwargs (dict, optional): kwargs for the plt.plot of the lines if draw_lines is True.
        **kwargs (): kwargs for the plt.text() call.
    """
    t0 = time.time()
    fig = ax.get_figure()
    if kwargs.get("transform", None) is None:
        fig.draw_without_rendering()
    else:
        if plot_kwargs is None:
            plot_kwargs = {}
        plot_kwargs["transform"] = kwargs.get("transform")
    aspect_ratio = fig.get_size_inches()[0] / fig.get_size_inches()[1]
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    xlims, ylims = data_to_display(xlims, ylims, ax)

    # Ensure good inputs and transform inputs
    assert len(x) == len(y)
    x = np.array(x)
    y = np.array(y)
    if scatter_sizes is not None:
        scatter_sizes = np.array(scatter_sizes)
    if text_scatter_sizes is not None:
        text_scatter_sizes = np.array(text_scatter_sizes)
    if x_scatter is not None:
        assert y_scatter is not None
    if y_scatter is not None:
        assert x_scatter is not None
        assert len(y_scatter) == len(x_scatter)
        x_scatter, y_scatter = data_to_display(
            x_scatter, y_scatter, ax, transform=kwargs.get("transform", None)
        )
    if x_lines is not None:
        assert y_lines is not None
    if y_lines is not None:
        assert x_lines is not None
        assert all(
            [len(x_line) == len(y_line) for x_line, y_line in zip(x_lines, y_lines)]
        )
        x_lines_temp, y_lines_temp = [], []
        for x_line, y_line in zip(x_lines, y_lines):
            xl, yl = data_to_display(x_line, y_line, ax)
            x_lines_temp.append(xl)
            y_lines_temp.append(yl)
        x_lines, y_lines = x_lines_temp, y_lines_temp
    assert min_distance <= max_distance
    if type(textsize) is not int:
        assert len(textsize) == len(x)
    else:
        textsize = [textsize for _ in range(len(x))]
    if type(textcolor) is not str:
        assert len(textcolor) == len(x)
    else:
        textcolor = [textcolor for _ in range(len(x))]
    if type(linecolor) is not str:
        assert len(linecolor) == len(x)
    else:
        linecolor = [linecolor for _ in range(len(x))]
    assert direction in [
        None,
        "south",
        "north",
        "east",
        "west",
        "southeast",
        "southwest",
        "northeast",
        "northwest",
    ]

    # Seed
    if seed > 0:
        randinds = np.arange(x.shape[0])
        np.random.seed(seed)
        np.random.shuffle(randinds)
        text_list = [text_list[i] for i in randinds]
        x = x[randinds]
        y = y[randinds]
        if text_scatter_sizes is not None:
            text_scatter_sizes = text_scatter_sizes[randinds]
        textsize = [textsize[i] for i in randinds]
        textcolor = [textcolor[i] for i in randinds]
        linecolor = [linecolor[i] for i in randinds]

    # Create boxes in original plot
    if verbose:
        print("Creating boxes")
    original_boxes = []
    for x_coord, y_coord, s, ts in tqdm(
        zip(x, y, text_list, textsize), disable=not verbose
    ):
        ann = ax.text(x_coord, y_coord, s, size=ts, **kwargs)
        box = ann.get_tightbbox(fig.canvas.get_renderer())
        w, h = box.x1 - box.x0, box.y1 - box.y0
        x_, y_ = data_to_display(
            [x_coord], [y_coord], ax, transform=kwargs.get("transform", None)
        )
        original_boxes.append((x_[0], y_[0], w, h, s))
        ann.remove()

    # Transform datapoints as well
    x, y = data_to_display(x, y, ax, transform=kwargs.get("transform", None))

    # If scatterplot exists, get scatter bboxes
    scatter_plot_bbs = None
    if scatter_plot is not None:
        scatter_plot_bbs = get_scatter_bbs(scatter_plot, ax)
        scatter_plot_bbs[:, 2] = scatter_plot_bbs[:, 0] + scatter_plot_bbs[:, 2]
        scatter_plot_bbs[:, 3] = scatter_plot_bbs[:, 1] + scatter_plot_bbs[:, 3]

    # Process extracted textboxes
    if verbose:
        print("Processing")
    if x_scatter is None:
        scatterxy = None
    else:
        scatterxy = np.transpose(np.vstack([x_scatter, y_scatter])).astype(np.float64)
    if x_lines is None:
        lines_xyxy = None
    else:
        lines_xyxy = lines_to_segments(x_lines, y_lines)

    non_overlapping_boxes, overlapping_boxes_inds = get_non_overlapping_boxes(
        original_boxes,
        xlims,
        ylims,
        aspect_ratio,
        margin,
        min_distance,
        max_distance,
        verbose,
        nbr_candidates,
        draw_all,
        scatterxy,
        lines_xyxy,
        scatter_sizes,
        scatter_plot_bbs,
        text_scatter_sizes,
        direction,
        draw_lines,
        avoid_label_lines_overlap,
    )

    # Plot once again
    if verbose:
        print("Plotting")
    if len(non_overlapping_boxes) == 0:
        if direction is not None:
            print(f"No non overlapping boxes found in direction {direction}")
        else:
            print("No non overlapping boxes found")
    if draw_lines:
        for x_coord, y_coord, w, h, s, ind in non_overlapping_boxes:
            x_near, y_near = find_nearest_point_on_box(
                x_coord, y_coord, w, h, x[ind], y[ind]
            )
            if x_near is not None:
                x_, y_ = display_to_data(
                    [x_near, x[ind]],
                    [y_near, y[ind]],
                    ax,
                    transform=kwargs.get("transform", None),
                )
                ax.plot(
                    x_,
                    y_,
                    linewidth=linewidth,
                    c=linecolor[ind],
                    **(plot_kwargs if plot_kwargs is not None else {}),
                )
    for x_coord, y_coord, w, h, s, ind in non_overlapping_boxes:
        if kwargs.get("ha", None) is not None:
            if kwargs.get("ha") == "center":
                x_coord += w / 2
            elif kwargs.get("ha") == "right":
                x_coord += w
        x_coord, y_coord = display_to_data(
            [x_coord], [y_coord], ax, transform=kwargs.get("transform", None)
        )
        ax.text(
            x_coord[0], y_coord[0], s, size=textsize[ind], c=textcolor[ind], **kwargs
        )

    if draw_all:
        for ind in overlapping_boxes_inds:
            x_coord, y_coord = display_to_data(
                [x[ind]], [y[ind]], ax, transform=kwargs.get("transform", None)
            )
            ax.text(
                x_coord[0],
                y_coord[0],
                text_list[ind],
                size=textsize[ind],
                c=textcolor[ind],
                **kwargs,
            )

    if verbose:
        print(f"Finished in {time.time()-t0}s")


def allocate_text(
    fig,
    ax,
    x: Union[np.ndarray, List[float]],
    y: Union[np.ndarray, List[float]],
    text_list: List[str],
    **kwargs,
):
    warnings.warn(
        "Usage of allocate_text will be replaced with allocate in future releases, removing the need for the fig argument"
    )
    allocate(ax, x, y, text_list, **kwargs)


def lines_to_segments(
    x_lines: List[np.ndarray],
    y_lines: List[np.ndarray],
) -> np.ndarray:
    """Sets up

    Args:
        x_lines (List[np.ndarray]): x-coordinates of all lines in plot list of 1d arrays
        y_lines (List[np.ndarray]): y-coordinates of all lines in plot list of 1d arrays

    Returns:
        np.ndarray: 2d array of line segments
    """
    assert len(x_lines) == len(y_lines)
    n_x_segments = int(np.sum([len(line_x) - 1 for line_x in x_lines]))
    n_y_segments = int(np.sum([len(line_y) - 1 for line_y in y_lines]))
    assert n_x_segments == n_y_segments
    lines_xyxy = np.zeros((n_x_segments, 4)).astype(np.float64)
    iter = 0
    for line_x, line_y in zip(x_lines, y_lines):
        for i in range(len(line_x) - 1):
            lines_xyxy[iter, :] = [line_x[i], line_y[i], line_x[i + 1], line_y[i + 1]]
            iter += 1
    return lines_xyxy


def data_to_display(x, y, ax, transform=None):
    """Transforms x and y in data coordinates to display coordinates."""
    xtrans, ytrans = [], []
    for x_, y_ in zip(x, y):
        if transform is not None:
            p = transform._as_mpl_transform(ax).transform_point((x_, y_))
        else:
            p = ax.transData.transform((x_, y_))
        xtrans.append(p[0])
        ytrans.append(p[1])
    return np.array(xtrans), np.array(ytrans)


def display_to_data(x, y, ax, transform=None):
    """Transforms x and y in display coordinates to data coordinates."""
    xtrans, ytrans = [], []
    for x_, y_ in zip(x, y):
        if transform is not None:
            p = transform._as_mpl_transform(ax).inverted().transform_point((x_, y_))
        else:
            p = ax.transData.inverted().transform((x_, y_))
        xtrans.append(p[0])
        ytrans.append(p[1])
    return np.array(xtrans), np.array(ytrans)


def get_scatter_bbs(sc, ax) -> np.ndarray:
    """Gets the bounding boxes of objects in a scatter plot

    Args:
        sc (): scatter plot object
        ax (): pyplot ax

    Returns:
        np.ndarray: 2d array of bounding boxes
    """
    ax.figure.canvas.draw()
    transform = sc.get_transform()
    transOffset = sc.get_offset_transform()
    offsets = sc._offsets
    paths = sc.get_paths()
    transforms = sc.get_transforms()

    if not transform.is_affine:
        paths = [transform.transform_path_non_affine(p) for p in paths]
        transform = transform.get_affine()
    if not transOffset.is_affine:
        offsets = transOffset.transform_non_affine(offsets)
        transOffset = transOffset.get_affine()
    if isinstance(offsets, np.ma.MaskedArray):
        offsets = offsets.filled(np.nan)

    bboxes = []
    if len(paths) and len(offsets):
        if len(paths) < len(offsets):
            paths = [paths[0]] * len(offsets)
        if len(transforms) < len(offsets):
            transforms = [transforms[0]] * len(offsets)
        for p, o, t in zip(paths, offsets, transforms):
            result = get_path_collection_extents(
                transform.frozen(), [p], [t], [o], transOffset.frozen()
            )
            bboxes.append(result.bounds)

    return np.array(bboxes)
