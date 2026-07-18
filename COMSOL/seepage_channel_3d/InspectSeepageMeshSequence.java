import com.comsol.model.*;
import com.comsol.model.util.*;
import java.util.Arrays;

/** Read-only diagnostic for the canonical source model mesh sequence. */
public class InspectSeepageMeshSequence {
  public static void main(String[] args) throws Exception {
    String inputModel = System.getenv("ATEM3D_COMSOL_INPUT_MODEL");
    if (inputModel == null || inputModel.trim().isEmpty()) {
      throw new IllegalArgumentException("missing ATEM3D_COMSOL_INPUT_MODEL");
    }
    ModelUtil.initStandalone(false);
    Model model = ModelUtil.load("inspect_seepage_mesh_sequence", inputModel);
    System.out.println("MESH_FEATURE_TAGS=" + Arrays.toString(
        model.component("comp1").mesh("mesh1").feature().tags()));
    for (String tag : model.component("comp1").mesh("mesh1").feature().tags()) {
      System.out.println("MESH_FEATURE=" + tag + " TYPE="
          + model.component("comp1").mesh("mesh1").feature(tag).getType());
    }
  }
}
