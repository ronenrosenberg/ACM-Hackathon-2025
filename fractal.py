import numpy as np
import pygame
import sounddevice as sd
import random
import math

#parallelization and jit compilation
from numba import njit
from numba import prange

# pygame setup
pygame.init()
#finds fullscreen resolution
screen_info = pygame.display.Info()
width, height = 1000, 1000
#creates our "display surface" with some useful parameters
flags = pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.HWACCEL
screen = pygame.display.set_mode((width, height), flags)
pygame.display.set_caption("Fractals!")

pygame.font.init()
default_font = pygame.font.SysFont('Roboto', 40)
win_font = pygame.font.SysFont('Roboto', 80)

#pygame clock
clock = pygame.time.Clock()

#audio intialization
current_freq = 0
phase = 0
def audio_callback(outdata, frames, time, status):
    global phase, current_freq
    t = (np.arange(frames) + phase) / 48000
    wave = np.sin(2*np.pi * current_freq * t)
    outdata[:] = wave.reshape(-1, 1) #need to reshape to 2D, even though only mono
    phase += frames
stream = sd.OutputStream(callback=audio_callback, samplerate=48000, channels=1, blocksize=1024)
stream.start()

#starting window size, max iterations for precision, and the frequency you're trying to find
x_min, x_max = -2.0,  1.0
y_min, y_max = -1.5,  1.5
max_iterations = 2000
target_freq = random.randrange(100, 1000, 50)


def detect_first_repeat(orbits):
    n = len(orbits)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = orbits[i]
            x2, y2 = orbits[j]
            dist = math.dist(orbits[i], orbits[j])
            if dist < 1e-4:
                return j - i  # peri
    return 0  # no repeat found


#proportionally maps a value from one range to another, ex: (0.5, [0, 1], [0, 100]) = 50
@njit
def map_value(x, in_range, out_range):
    return out_range[0] + (x - in_range[0]) * (out_range[1] - out_range[0]) / (in_range[1] - in_range[0])

@njit(parallel=True) #jit compilation magic that uses parallel processing
def mandelbrot(x_min, x_max, y_min, y_max):
    c_iter_counts = np.empty((height, width), dtype=np.uint16)

    #for each pixel
    for y in prange(height):
        ci = map_value(y, (0, height-1), (y_min, y_max)) #imaginary part of c (using real math to emulate the imaginary part of this turns out to be faster)
        for x in range(width):
            cr = map_value(x, (0, width-1), (x_min, x_max)) #real part of c
            
            zr, zi = 0, 0
            prev_zr, prev_zi = 0, 0

            iter_count = 1
            while True:
                #checks if radius > 2 (at which point it's proven it will zoom to infinity)
                if zr*zr + zi*zi > 4 or iter_count == max_iterations: # if escapes, break
                    c_iter_counts[y, x] = iter_count
                    break

                #optimization, catches likely convergence or cycle early 
                if iter_count % 10 == 0:
                    if abs(zr - prev_zr) < 1e-12 and abs(zi - prev_zi) < 1e-12:
                        c_iter_counts[y, x] = max_iterations
                        break
                    prev_zr = zr
                    prev_zi = zi
                
                zi, zr = 2 * zr * zi + ci, zr**2 - zi**2 + cr
                iter_count += 1
    return c_iter_counts

def colorize(c_iter_counts):
    #change all 0s to 1 to avoid log(0), undefined
    c_iter_counts = np.where(c_iter_counts == 0, 1, c_iter_counts)

    norm = np.log(c_iter_counts) / np.log(max_iterations)
    gray = np.abs((norm * 255).astype(np.uint8) - 255)
    #(gray**2, gray*-0.3, gray*5)
    colors = np.stack((gray**2, gray*.2, gray**1.1), axis=-1) * 2
    return colors

def render_mandelbrot():
    c_iter_counts = mandelbrot(x_min, x_max, y_min, y_max)
    colors = colorize(c_iter_counts)

    #transpose changes axes so instead of (height, width, 3) is (width, height, 3)
    return pygame.surfarray.make_surface(np.transpose(colors, (1, 0, 2)))


fractal = render_mandelbrot()
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        elif event.type == pygame.KEYDOWN:
            x_zoom = (x_max - x_min)/8
            y_zoom = (y_max - y_min)/8
            if event.key == pygame.K_f:
                x_min += x_zoom
                x_max -= x_zoom
                y_min += y_zoom
                y_max -= y_zoom
            if event.key == pygame.K_g:
                x_min -= x_zoom
                x_max += x_zoom
                y_min -= y_zoom
                y_max += y_zoom
            elif event.key == pygame.K_UP:
                y_max -= y_zoom
                y_min -= y_zoom
            elif event.key == pygame.K_DOWN:
                y_max += y_zoom
                y_min += y_zoom
            elif event.key == pygame.K_LEFT:
                x_max -= x_zoom
                x_min -= x_zoom
            elif event.key == pygame.K_RIGHT:
                x_max += x_zoom
                x_min += x_zoom

            #rerender fractal
            fractal = render_mandelbrot()
        elif event.type == pygame.MOUSEBUTTONUP:
            current_freq = 0
                
    screen.blit(fractal, (0, 0))      
    if pygame.mouse.get_pressed()[0]:
        (x, y) = pygame.mouse.get_pos()
        ci = map_value(y, (0, height-1), (y_min, y_max))
        cr = map_value(x, (0, width-1), (x_min, x_max))
        
        zr, zi = 0, 0
        prev_zi, prev_zr = 0, 0

        coord_list = [(x, y)]
        z_list = [(zi, zr)]

        iter_count = 1
        while True:
            #checks if radius > 2 (at which point it's proven it will zoom to infinity)
            if zr**2 + zi**2 > 4 or iter_count == max_iterations:
                break
            
            #optimization, catches likely convergence or cycle early 
            if iter_count % 10 == 0:
                if abs(zr - prev_zr) < 1e-12 and abs(zi - prev_zi) < 1e-12:
                    break
                prev_zr = zr
                prev_zi = zi

            zi, zr = 2 * zr * zi + ci, zr**2 - zi**2 + cr
            #update z list and corresponding element of coord list   
            z_list.append((zi, zr))
            new_x, new_y = int(np.interp(zr, (x_min, x_max), (0, width-1))) , int(np.interp(zi, (y_min, y_max), (0, height-1)))
            coord_list.append((new_x, new_y))
            
            iter_count += 1

        pygame.draw.lines(screen, (255, 255, 255), False, coord_list, width=1)
        period = detect_first_repeat(z_list)
        print(period)
        current_freq = 50 * period

    #display target and current frequencies
    target_frequency_text = default_font.render(f"Target frequency: {target_freq}", True, (255, 255, 255))
    current_frequency_text = default_font.render(f"Current frequency: {current_freq}", True, (255, 255, 255)) 
    screen.blit(target_frequency_text, (4, height-60))
    screen.blit(current_frequency_text, (4, height-30))

    #win condition
    if current_freq == target_freq:
        win_text = win_font.render(f"You found the frequency!", True, (255, 255, 255))
        screen.blit(win_text, (width/12, height/2-height/20))
    
    pygame.display.flip()
    clock.tick(60)
