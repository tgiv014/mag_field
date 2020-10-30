# mag_field
This project draws pretty pictures of magnetic fields.

![Three Examples](static/threefold_small.png)

# Method
1. Place infinitely long wires perpendicular to the image plane at random points.
2. Pick random currents to flow through those wires.
3. Calculate the resulting magnetic field across the image plane.
4. Seperately calculate a similar (magnitude only) field used for color mapping.
5. Generate a list of points to start drawing streamlines from.
6. Sort that list by the colormap value of each startpoint.
7. Render all the streamlines!
