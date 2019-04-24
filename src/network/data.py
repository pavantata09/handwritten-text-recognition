# def to_sparse(texts):
#     """Put ground truth texts into sparse tensor for ctc_loss"""

#     shape = [len(texts), 0]
#     indices, values = [], []

#     for (batch_element, text) in enumerate(texts):
#         label_str = [env.CHAR_LIST.index(c) for c in text]

#         if len(label_str) > shape[1]:
#             shape[1] = len(label_str)

#         for (i, label) in enumerate(label_str):
#             indices.append([batch_element, i])
#             values.append(label)

#     return np.array((indices, values, shape))


# def to_text(ctc_output):
#     """Extract texts from output of CTC decoder"""

#     encoded_label_strs = [[] for i in range(model.BATCH_SIZE)]
#     decoded = ctc_output[0][0]

#     for (idx, idx2d) in enumerate(decoded.indices):
#         label = decoded.values[idx]
#         batch_element = idx2d[0]
#         encoded_label_strs[batch_element].append(label)

#     return [str().join([env.CHAR_LIST[c] for c in label_str]) for label_str in encoded_label_strs]


# def data_generator(partition, preproc, ground_truth):

#     with open(partition, "r") as file:
#         data_list = [x.strip() for x in file.readlines()]

#     for item in data_list:
#         data = np.load(os.path.join(preproc, f"{item}.npy"))
#         data = np.reshape(data, data.shape + (1,))
#         data = np.reshape(data, (1,) + data.shape)

#         with open(os.path.join(ground_truth, f"{item}.txt"), "r") as file:
#             lines = [x.strip() for x in file.readlines()]
#             target_data = np.array([' '.join(lines)])

#             # target_data = np.reshape(target_data, target_data.shape + (1,))

#             # print("\n\n", target_data, "\n\n")
#             # target_data = to_sparse(target_data)
#             # print(temp, "\n\n")

#             target_data = np.reshape(target_data, (1,) + target_data.shape)
#             print(target_data.shape)

#             # target_data = np.reshape(target_data, target_data.shape + (1,))

#         yield ({"the_inputs": data,
#                 "the_labels": target_data,
#                 "input_length": np.array([len(data[0][0]) * len(data[0][1])]),
#                 "label_length": np.array([[1]])
#                 }, {'ctc': target_data})