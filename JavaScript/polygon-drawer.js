/* Adapted and extended from:
  How to draw any regular shape with just one JavaScript function - https://developer.mozilla.org/en-US/blog/javascript-shape-drawing-function/ */

// Function to draw any polygon
const drawShape = (context, x, y, r, sides) => {
  // Move the canvas to the centre position
  context.translate(x, y);

  for (let i = 0; i < sides; i++) {
    // Calculate the rotation
    const rotation = ((Math.PI * 2) / sides) * i;

    // For the first point move to
    if (i === 0) {
      context.moveTo(r * Math.cos(rotation), r * Math.sin(rotation));
    } else {
      // For the rest draw a line
      context.lineTo(r * Math.cos(rotation), r * Math.sin(rotation));
    }
  }
  // Close path and stroke it
  context.closePath();
  context.stroke();

  // Reset the translate position
  context.resetTransform();
}

export default drawShape;