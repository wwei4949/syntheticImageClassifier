import csv
import os
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


if __name__ == "__main__":
    with open("model/results.pkl", "rb") as f:
        test_results = pickle.load(f)

    real_names = ["0354892760101035_175780808_frontdrivercorner_Claims1-50B2B380-0000-C71D-9CCE-450B2E629878.jpg",
                  "0608561990101074___905310005745.jpg",
                  "8677738340000002_171449228_backdrivercorner_Claims1-00BA0C7E-0000-C819-BF68-15DDE7FAF9F5.jpg",
                  "0418711480101013_175743918_frontpassengercorner_Claims1-0071A580-0000-C511-A07C-1F783552B632.jpg",
                  "0684241500000001_173362567_frontpassengercorner_Claims1-10ED327F-0000-CA36-A922-9DDA7CC622AC.jpg",
                  "0528985010000005_177873531_backpassengercorner_Claims2-604DC481-0000-C512-B20F-F3AED3BF4173.jpg",
                  "8687280560000001___905291649678.jpg",
                  "8756008890000001_176887136_backdrivercorner_Claims2-80D64081-0000-CA1B-9C6F-BC5766EFA69E.jpg",
                  "8697696940000001___Claims1-70579079-0000-CE28-8EFC-735F076FD444.jpg",
                  "0584491750000003_172068515_frontpassengercorner_Claims1-103C6F7E-0000-C03A-8011-4A18845D2244.jpg",
                  "0585831180101172___905277056017.jpg", "0435101100101031___905288846854.jpg",
                  "8709426220000001_179343450_frontpassengercorner_Claims2-F0949782-0000-C915-85F0-53D8F9D266D5.jpg",
                  "0592133610101020___Claims1-B085D278-0000-C52F-886E-C11CEF610A81.jpg",
                  "0395030360101043___905251888079.jpg",
                  "8701563650000001_175160890_frontdrivercorner_Claims1-506F4780-0000-C312-88D1-FFB2F3492A41.jpg",
                  "8668873630000001___905252218353.jpg", "8675484330000001___905263690132.jpg"]
    generated_names = ["105_01.png", "002_02.png", "004_00.png", "100_00.png", "111_00.png", "091_01.png", "125_01.png",
                       "144_00.png", "039_02.png", "057_01.png", "092_00.png", "024_01.png", "247_00.png", "018_02.png",
                       "076_02.png", "044_00.png", "171_00.png", "080_00.png", "161_01.png", "158_00.png", "234_01.png",
                       "115_00.png", "055_02.png", "106_02.png", "116_00.png"]
    inpainted_names = ["049_intensely_damaged_bumper.png", "355_extraordinarily_severed_vehicle.png",
                       "330_hugely_disintegrated_bumper.png", "216_severely_fractured_vehicle.png",
                       "040_overly_crushed_bumper.png", "058_very_fragmented_bumper.png",
                       "091_highly_crushed_glass.png", "119_extraordinarily_crumbled_bumper.png",
                       "129_hugely_fragmented_bumper.png", "179_intensely_collapsed_bumper.png",
                       "351_severely_shattered_bumper.png", "400_very_mutilated_bumper.png",
                       "261_terrifically_mangled_bumper.png", "190_acutely_demolished_glass.png",
                       "047_overly_mangled_vehicle.png", "077_exceedingly_smashed_vehicle.png",
                       "035_hugely_damaged_vehicle.png", "021_overly_shattered_glass.png",
                       "348_hugely_mutilated_bumper.png", "385_acutely_demolished_bumper.png",
                       "469_severely_demolished_bumper.png", "001_immensely_collapsed_bumper.png",
                       "156_utterly_crumbled_vehicle.png", "582_terrifically_smashed_vehicle.png",
                       "613_exceedingly_crumbled_glass.png"]
    all_names = real_names + generated_names + inpainted_names

    correct_dict = {   # keeping track of both correct & incorrect just as a sanity check (even though we know sum)
        "real":      {"correct": 0, "incorrect": 0},
        "generated": {"correct": 0, "incorrect": 0},
        "inpainted": {"correct": 0, "incorrect": 0}
    }
    gts, preds = [], []
    for test_result in test_results:
        gt_label = int(test_result["gt_label"])
        pred = int(test_result["pred_label"])
        img_name = os.path.basename(test_result["img_path"])
        if img_name in all_names:
            gts.append(gt_label)
            preds.append(pred)
        for key, names in zip(("real", "generated", "inpainted"), (real_names, generated_names, inpainted_names)):
            if img_name in names:
                correct_dict[key]["correct" if gt_label == pred else "incorrect"] += 1
                break
    print(f"n={len(gts)}")
    print(correct_dict)

    acc = accuracy_score(y_true=gts, y_pred=preds)
    recall = recall_score(y_true=gts, y_pred=preds)
    conf_mat = confusion_matrix(y_true=gts, y_pred=preds)
    prec = precision_score(y_true=gts, y_pred=preds)
    print(f"accuracy: {acc:.2f}")
    print(f"precision: {prec:.2f}")
    print(f"recall: {recall:.2f}")
    print(f"confusion:\n{conf_mat}")

    # write csv
    mapping = {0: "REAL", 1: "FAKE"}
    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        field = ["img_path", "gt_label", "pred_label", "gt_name", "pred_name"]
        writer.writerow(field)
        for test_result in test_results:
            gt_label = int(test_result["gt_label"])
            pred = int(test_result["pred_label"])
            writer.writerow([test_result["img_path"], gt_label, pred, mapping[gt_label], mapping[pred]])
