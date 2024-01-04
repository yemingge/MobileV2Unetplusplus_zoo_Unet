# import matplotlib.pyplot as plt
#
# data = [['Moedels', 'Flops(G)', 'Params(M)'],
#         ['Unet', 72.528, 34.526],
#         ['Unet++', 153.059, 36.628],
#         ['MobileUnet', 45.886, 15.112],
#         ['Resunet++', 17.933, 4.063],
#         ['MobileV2Unet++(ours)', '10.138'+"√", '3.182√'],]
#
# fig, ax = plt.subplots()
# table = plt.table(cellText=data, loc='center', cellLoc='center', colLabels=None, cellColours=None)

# ax.axis('off')

# plt.show()
import matplotlib.pyplot as plt

data = [['Moedels', 'runtime(s)'],
        ['Unet', 0.472],
        ['Unet++', 1.038],
        ['MobileUnet', 0.262],
        ['Resunet++', 0.379],
        ['MobileV2Unet++(ours)', '0.246√'],]

fig, ax = plt.subplots()
table = plt.table(cellText=data, loc='center', cellLoc='center', colLabels=None, cellColours=None)

ax.axis('off')

plt.show()