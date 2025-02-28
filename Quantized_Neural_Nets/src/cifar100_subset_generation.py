import numpy as np

dist_matrix = np.load("./cifar100_kl_div_matrix.npy")
class_names = np.load("./cifar100_class_names.npy")

def generate_subset(n_classes, dist_matrix_path = "./cifar100_kl_div_matrix.npy", class_names_path = "./cifar100_class_names.npy", similar_classes = True):

    assert n_classes > 1

    kls_cf_matrix = np.load(dist_matrix_path)
    class_names = np.load(class_names_path)

    if similar_classes is not None:
        
        rand_class_ind = np.random.randint(0, class_names.shape[0])
        rand_class_name = class_names[rand_class_ind]

        selected_classes = {rand_class_ind}
        for _ in range(n_classes - 1):
            if similar_classes:
                sorted_classes = np.argsort(kls_cf_matrix[list(selected_classes)].mean(axis = 0))
            else:
                sorted_classes = np.argsort(kls_cf_matrix[list(selected_classes)].mean(axis = 0))[::-1]
            i = 0
            while sorted_classes[i] in selected_classes:
                i += 1
            selected_classes.add(sorted_classes[i].item())
        
        selected_classes = list(selected_classes)
    else:
        selected_classes = np.random.default_rng().choice(class_names.shape[0], size = n_classes, replace = False)
    
    subset_dists = []
    subset_classes = []
    for i in range(len(selected_classes)):
        c = selected_classes[i]
        subset_classes += [c]
        for j in range(i + 1, len(selected_classes)):
            subset_dists += [kls_cf_matrix[selected_classes[i], selected_classes[j]]]
    
    return {"classes": subset_classes, "max_dist": max(subset_dists), "min_dist": min(subset_dists), "avg_dist": np.mean(subset_dists), "median_dist": np.median(subset_dists)}