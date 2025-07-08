#!/usr/bin/env python3
#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Tue Jul  8 11:03:35 2025
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.text import Text


class DraggableText:
    """
    A class to make a matplotlib.text.Text object draggable.
    Leverages Matplotlib's event handling system and blitting for smooth interaction.
    """
    lock = None # Ensures only one text is draggable at a time

    def __init__(self, text_artist: Text):
        if not isinstance(text_artist, Text):
            raise TypeError("DraggableText must be initialized with a matplotlib.text.Text object.")
        self.text_artist = text_artist
        # This will store the initial text and mouse coordinates as a flat tuple
        # (initial_text_x, initial_text_y, initial_mouse_x, initial_mouse_y)
        self.press_data = None
        self.background = None

    def connect(self):
        """Connect to all the events we need for dragging."""
        self.cidpress = self.text_artist.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.text_artist.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.text_artist.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        # Optional: Re-capture background on full redraws (e.g., window resize)
        self.cid_draw = self.text_artist.figure.canvas.mpl_connect(
            'draw_event', self.on_draw)

    def on_draw(self, event):
        """Callback to re-capture the background after a full draw."""
        # We only need to save the background if a drag is in progress
        if self.press_data is not None:
            self.background = self.text_artist.figure.canvas.copy_from_bbox(
                self.text_artist.axes.bbox)

    def on_press(self, event):
        """Callback for button press event."""
        if (event.inaxes != self.text_artist.axes or DraggableText.lock is not None):
            return

        contains, _ = self.text_artist.contains(event)
        if not contains:
            return

        # Store the initial position of the text and the initial position of the mouse
        x_text_orig, y_text_orig = self.text_artist.get_position()
        x_mouse_orig, y_mouse_orig = event.xdata, event.ydata
        
        # Store all four scalar values in a single tuple for easy unpacking later
        self.press_data = (x_text_orig, y_text_orig, x_mouse_orig, y_mouse_orig)
        
        DraggableText.lock = self

        canvas = self.text_artist.figure.canvas
        self.background = canvas.copy_from_bbox(self.text_artist.axes.bbox)
        self.text_artist.set_animated(True) # Makes drawing faster during drag
        self.text_artist.draw(canvas.renderer)
        canvas.blit(self.text_artist.axes.bbox)

    def on_motion(self, event):
        """Callback for mouse motion event (dragging)."""
        if (self.press_data is None or
                event.inaxes != self.text_artist.axes or
                DraggableText.lock is not self):
            return

        # Unpack the stored initial positions
        x_text_orig, y_text_orig, x_mouse_orig, y_mouse_orig = self.press_data

        # Calculate the displacement of the mouse from its initial press position
        dx = event.xdata - x_mouse_orig
        dy = event.ydata - y_mouse_orig

        # Calculate the new position for the text
        new_x = x_text_orig + dx
        new_y = y_text_orig + dy

        self.text_artist.set_position((new_x, new_y))

        canvas = self.text_artist.figure.canvas
        axes = self.text_artist.axes

        # Restore the saved background, draw the animated text, and blit
        canvas.restore_region(self.background)
        self.text_artist.draw(canvas.renderer)
        canvas.blit(axes.bbox)

    def on_release(self, event):
        """Callback for button release event."""
        if DraggableText.lock is not self:
            return

        self.press_data = None # Clear stored data
        DraggableText.lock = None # Release the lock
        self.background = None # Clear background
        self.text_artist.set_animated(False) # Turn off animation
        self.text_artist.figure.canvas.draw() # Force full redraw

    def disconnect(self):
        """Disconnect all connected events."""
        self.text_artist.figure.canvas.mpl_disconnect(self.cidpress)
        self.text_artist.figure.canvas.mpl_disconnect(self.cidrelease)
        self.text_artist.figure.canvas.mpl_disconnect(self.cidmotion)
        self.text_artist.figure.canvas.mpl_disconnect(self.cid_draw)

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(np.random.rand(10), np.random.rand(10), 'o', alpha=0.7)
    ax.set_title("Drag the text objects!")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Create multiple text artists
    text1 = plt.text(0.2, 0.8, 'Drag Me 1', # Source
                           fontsize=18, ha='center', va='center', color='red',
                           bbox=dict(facecolor='pink', alpha=0.7, pad=5),
                           zorder=10)
                           
    text2 = plt.text(0.7, 0.3, 'Drag Me 2', # Source
                           fontsize=16, ha='center', va='center', color='green',
                           bbox=dict(facecolor='lightgreen', alpha=0.7, pad=5),
                           zorder=10)

    text3 = plt.text(0.5, 0.5, 'Drag Me 3', # Source
                           fontsize=20, ha='center', va='center', color='blue',
                           bbox=dict(facecolor='lightblue', alpha=0.7, pad=5),
                           zorder=10)

    # Create a DraggableText instance for each text artist and connect them
    draggable_text_instance1 = DraggableText(text1)
    draggable_text_instance1.connect()

    draggable_text_instance2 = DraggableText(text2)
    draggable_text_instance2.connect()

    draggable_text_instance3 = DraggableText(text3)
    draggable_text_instance3.connect()

    plt.show()
