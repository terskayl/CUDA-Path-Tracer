import struct

def float_to_rgbe(r, g, b):
    """
    Converts floating-point RGB values to the 4-byte RGBE format.
    The formula is: rgbe = (mantissa, mantissa, mantissa, exponent)
    """
    # Calculate the maximum component value
    v = max(r, g, b)
    if v < 1e-32:
        return 0, 0, 0, 0
    
    # Calculate the exponent and mantissas
    # The exponent is base 2, but for RGBE it is encoded as exponent + 128
    
    # Determine the exponent value
    exponent = int(v.as_integer_ratio()[1].bit_length() - 1)
    
    # This is a bit of a hack, but the RGBE formula is
    # mantissa = (value * 256) / (2^exponent)
    # The `math.frexp` function from the math module is better for this.
    # We will compute it manually here for demonstration.
    
    # Find the smallest power of 2 greater than or equal to the max value
    import math
    v, exponent = math.frexp(v)
    
    v = v * 256.0 / 256.0 # Normalize the mantissas
    
    # Scale the RGB values based on the exponent
    scale = 255.0 * (v / max(r, g, b))
    
    r_mantissa = int(r * scale)
    g_mantissa = int(g * scale)
    b_mantissa = int(b * scale)
    
    return r_mantissa, g_mantissa, b_mantissa, exponent + 128

def create_basic_hdr(filename):
    """
    Creates a basic 1x1 Radiance HDR file with a bright orange color.
    """
    # Floating-point color for a very bright orange
    r_float = 100.0  # High intensity red
    g_float = 50.0   # Medium intensity green
    b_float = 0.0    # No blue
    
    # Convert float RGB to RGBE
    r_rgbe, g_rgbe, b_rgbe, exponent = float_to_rgbe(r_float, g_float, b_float)

    # Header section
    header = "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 1\n"
    
    # Pixel data in RGBE format
    pixel_data = bytes([r_rgbe, g_rgbe, b_rgbe, exponent])

    # Write to file
    with open(filename, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(pixel_data)

    print(f"Successfully created '{filename}'")

if __name__ == "__main__":
    create_basic_hdr("basic_orange.hdr")