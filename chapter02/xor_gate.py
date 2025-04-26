from and_gate import and_gate
from nand_gate import nand_gate
from or_gate import or_gate

def xor_gate(x1, x2):
	s1 = nand_gate(x1, x2)
	s2 = or_gate(x1, x2)
	y = and_gate(s1, s2)
	return y

if __name__ == '__main__':
	for xs in [(0,0), (0,1), (1,0), (1,1)]:
		y = xor_gate(xs[0], xs[1])
		print(str(xs) + "-->" + str(y))