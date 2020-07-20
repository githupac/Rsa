# ---> 981813172
# >>> muhamd Ghavi

#random algorithm

import numpy as np
import time

def pseudo_uniform_good(mult=16807, mod=(2**23)-1, seed=123456789, size=1):
        u = np.zeros(size)
        x = (seed*mult+1)%mod
        u[0] = x/mod
        for i in range(1,size):
                x = (x*mult+1)%mod
                u[i] = x/mod
        return u
def pseudo_uniform(l=0, h=1, se= 123456789, si=1):
        return l+(h-l)* pseudo_uniform_good(seed=se, size=si)

def p_r (mu=0.0, sigma=1.0, size=1):
        t = time.perf_counter()
        seed1 = int(10**9*float(str(t-int(t))[0:]))
        u1 = pseudo_uniform(se=seed1, si=size)
        t = time.perf_counter()
        seed2 = int(10**9*float(str(t-int(t))[0:]))
        u2 = pseudo_uniform(se=seed2, si=size)

        z0= np.sqrt(-2*np.log(u1))*np.cos(2*np.pi*u2)
        z1 = np.sqrt(-2*np.log(u1))*np.cos(2*np.pi*u2)

        z0 = z0*sigma+mu
        
        return int(str(abs(z0[0]))[2:])




#miller_rabin algorithm
#ood or even

import random   # ------> only for miller_rabin algorithm
def miller_rabin(n, k=40):


    # If number is even, it's a composite number

    if n in  [2,3]:
        return True

    
    if n in [0,1]:
        return False


    if n % 2 == 0:
        return False

    r, s = 0, n - 1
    while s % 2 == 0:
        r += 1
        s //= 2
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, s, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True






#karatsuba algorithm
# Fast multiplication

import math


def karatsuba(x, y):
    if x < 10 and y < 10:
        return x * y

    n = max(len(str(x)), len(str(y)))
    m = int(math.ceil(float(n) / 2))

    # divide x into two half
    xh = int(math.floor(x / 10 ** m))
    xl = int(x % (10 ** m))

    # divide y into two half
    yh = int(math.floor(y / 10 ** m))
    yl = int(y % (10 ** m))

    # Karatsuba's algorithm.
    s1 = karatsuba(xh, yh)
    s2 = karatsuba(yl, xl)
    s3 = karatsuba(xh + xl, yh + yl)
    s4 = s3 - s2 - s1

    return int(s1 * (10 ** (m*2)) + s4 * (10 ** m) + s2)







#fast pow algorithm


def fast_exp(b, e, m):
    r = 1
    if 1 & e:
        r = b
    while e:
        e >>= 1
        b = (b * b) % m
        if e & 1: r = (r * b) % m
    return r








import random # ----> Not for p & q 


'''
Euclid's algorithm for determining the greatest common divisor
Use iteration to make it faster for larger integers
'''
def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

'''
Euclid's extended algorithm for finding the multiplicative inverse of two numbers
'''
def multiplicative_inverse(e, phi):
    d = 0
    x1 = 0
    x2 = 1
    y1 = 1
    temp_phi = phi
    
    while e > 0:
        temp1 = temp_phi//e
        temp2 = temp_phi - karatsuba(temp1, e)
        temp_phi = e
        e = temp2
        
        x = x2- karatsuba(temp1, x1)
        y = d - karatsuba(temp1, y1)
        
        x2 = x1
        x1 = x
        d = y1
        y1 = y
    if temp_phi == 1:
        return d + phi

'''
Tests to see if a number is prime.
'''


def generate_keypair(p, q):

    #n = pq

    n = karatsuba(p, q)

    #Phi is the totient of n
    phi = karatsuba((p-1), (q-1))


    #Choose an integer e such that e and phi(n) are coprime
    e = random.randrange(1, phi)


    #Use Euclid's Algorithm to verify that e and phi(n) are comprime
    g = gcd(e, phi)
    while g != 1:
        e = random.randrange(1, phi)
        g = gcd(e, phi)
   
    #Use Extended Euclid's Algorithm to generate the private key
    d = multiplicative_inverse(e, phi)
 
    #Return public and private keypair
    #Public key is (e, n) and private key is (d, n)
    return ((e, n), (d, n))

def encrypt(pk, plaintext):
    #Unpack the key into it's components
    key, n = pk
    
    #Convert each letter in the plaintext to numbers based on the character using a^b mod m
    cipher = [fast_exp(ord(char), key, n) for char in plaintext]
    #Return the array of bytes
    return cipher

def decrypt(pk, ciphertext):
    #Unpack the key into its components
    key, n = pk
    #Generate the plaintext based on the ciphertext and key using a^b mod m
    plain = []
    for char in ciphertext:
            nb = fast_exp(char, key, n)
            
            plain.append(chr(nb))
 
    #Return the array of bytes as a string
    return ''.join(plain)
  

       


if __name__ == '__main__':
    '''
    Detect if the script is being run directly by the user
    '''
    print()
    print("Welcome to the powerful, secure and ultra-fast encryption system RSA ")
    while True:
            p = p_r ()
            if miller_rabin(p):
                    break

 
    while True:
           q = p_r ()
           if miller_rabin(q) and q != p:
                   break

    print()
    public, private = generate_keypair(p, q)
    print("Your public key is = ", public)
    print("your private key is = ", private)
    print()


    while 1:
            aq = input("Which key do you want to use to lock the text? (public_key / private_key) :")
            print()
            try:
                    if aq in ['private_key', 'private']:
                            message = input(f"Enter a message to encrypt with your {aq}: ")
                            encrypted_msg = encrypt(private, message)
                            print()
                            print ("Your encrypted message is: ",''.join(map(lambda x: str(x), encrypted_msg)))
                  
                 
                            print()
                            print ("Decrypting message with public key. Your message is: ", decrypt(public, encrypted_msg))

                            print()
                            break


                    elif aq in ['public_key', 'public']:
                            message = input(f"Enter a message to encrypt with your {aq}: ")
                            encrypted_msg = encrypt(public, message)
                            print()
                            print ("Your encrypted message is: ", ''.join(map(lambda x: str(x), encrypted_msg)))
                  
                            print()
                            print ("Decrypting message with private key. Your message is: ", decrypt(private, encrypted_msg))

                            print()
                            break
                    else :
                           raise ValueError

            except ValueError:
                    print(f"Error....! What? {aq}?")
                    print()
            
    print("it's strong...!!!")
    print("author: Muhamad Ghavi")
