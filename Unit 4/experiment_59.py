weight=.5; derivative=.8
print("Time steps | Gradient magnitude")
for n in [1,2,3,5,10,20,30,40]:
 g=(weight*derivative)**n
 print(f"{n:10d} | {g:.10f}")
print("The gradient becomes very small over many time steps, making long-term learning difficult.")
