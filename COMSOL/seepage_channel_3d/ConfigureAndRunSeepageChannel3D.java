import com.comsol.model.*;
import com.comsol.model.util.*;
import java.util.Arrays;

/** Independent 60 x 1 x 1 m seepage-channel run derived from a read-only MPH. */
public class ConfigureAndRunSeepageChannel3D {
  private static final String TLIST =
      "0 1e-7 2e-7 5e-7 1e-6 2e-6 5e-6 "
          + "1e-5 1.2589254117941673e-5 1.584893192461114e-5 "
          + "1.9952623149688796e-5 2.5118864315095822e-5 3.1622776601683795e-5 "
          + "3.9810717055349695e-5 5.011872336272725e-5 6.309573444801929e-5 "
          + "7.943282347242822e-5 1e-4 1.2589254117941662e-4 "
          + "1.584893192461114e-4 1.9952623149688827e-4 2.511886431509582e-4 "
          + "3.1622776601683794e-4 3.981071705534969e-4 5.011872336272725e-4 "
          + "6.309573444801929e-4 7.943282347242822e-4 1e-3 "
          + "1.2589254117941661e-3 1.584893192461114e-3 1.9952623149688828e-3 "
          + "2.511886431509582e-3 3.1622776601683794e-3 3.981071705534969e-3 "
          + "5.011872336272725e-3 6.309573444801929e-3 "
          + "7.943282347242822e-3 1e-2";

  public static void main(String[] args) throws Exception {
    String inputModel = requiredEnv("ATEM3D_COMSOL_INPUT_MODEL");
    String outputModel = requiredEnv("ATEM3D_COMSOL_OUTPUT_MODEL");
    String outputCsv = requiredEnv("ATEM3D_COMSOL_OUTPUT_CSV");
    String caseName = requiredEnv("ATEM3D_COMSOL_CASE");
    String sigmaChannel = requiredEnv("ATEM3D_COMSOL_CHANNEL_SIGMA");
    if (inputModel.equalsIgnoreCase(outputModel)) {
      throw new IllegalArgumentException("input and output MPH must be distinct");
    }
    if (!(caseName.equals("background") || caseName.equals("zero_contrast")
        || caseName.equals("channel"))) {
      throw new IllegalArgumentException("unknown seepage case: " + caseName);
    }

    ModelUtil.initStandalone(false);
    ModelUtil.showProgress(true);
    Model model = ModelUtil.load("seepage_channel_3d_" + caseName, inputModel);
    model.param().set("sigma_channel", sigmaChannel + "[S/m]");
    model.param().set("eps_r", "0");

    removeChannelFeatures(model);
    createChannelGeometryAndSelection(model);
    createChannelMeshControl(model);
    if (!caseName.equals("background")) {
      createChannelMaterial(model);
      createChannelMefFeature(model);
    }

    configureReceivers(model, outputCsv);
    configureSolver(model);
    model.component("comp1").mesh("mesh1").run();
    System.out.println("SEEPAGE_CHANNEL_DOMAINS=" + Arrays.toString(
        model.component("comp1").selection("sel_channel_3d").entities(3)));

    model.sol("sol1").clearSolution();
    setStationaryConfiguration(model, !caseName.equals("background"));
    System.out.println("SEEPAGE_STAGE=stationary case=" + caseName);
    model.sol("sol1").runFromTo("st1", "su1");
    setTransientConfiguration(model, !caseName.equals("background"));
    System.out.println("SEEPAGE_STAGE=transient case=" + caseName);
    model.sol("sol1").runFromTo("st2", "t1");

    model.result().export("recv_csv").run();
    model.save(outputModel);
    System.out.println("SEEPAGE_OUTPUT_CSV=" + outputCsv);
    System.out.println("SEEPAGE_OUTPUT_MODEL=" + outputModel);
  }

  private static String requiredEnv(String name) {
    String value = System.getenv(name);
    if (value == null || value.trim().isEmpty()) {
      throw new IllegalArgumentException("missing environment variable " + name);
    }
    return value;
  }

  private static void createChannelGeometryAndSelection(Model model) {
    model.component("comp1").geom("geom1").create("blk_channel_3d", "Block");
    model.component("comp1").geom("geom1").feature("blk_channel_3d").set(
        "size", new String[] {"60[m]", "1[m]", "1[m]"});
    model.component("comp1").geom("geom1").feature("blk_channel_3d").set(
        "pos", new String[] {"-30[m]", "-0.5[m]", "19.5[m]"});
    model.component("comp1").geom("geom1").feature("blk_channel_3d").set("selresult", true);
    model.component("comp1").geom("geom1").run();

    model.component("comp1").selection().create("sel_channel_3d", "Box");
    model.component("comp1").selection("sel_channel_3d").set("entitydim", 3);
    model.component("comp1").selection("sel_channel_3d").set("condition", "inside");
    model.component("comp1").selection("sel_channel_3d").set("xmin", "-30.000001[m]");
    model.component("comp1").selection("sel_channel_3d").set("xmax", "30.000001[m]");
    model.component("comp1").selection("sel_channel_3d").set("ymin", "-0.500001[m]");
    model.component("comp1").selection("sel_channel_3d").set("ymax", "0.500001[m]");
    model.component("comp1").selection("sel_channel_3d").set("zmin", "19.499999[m]");
    model.component("comp1").selection("sel_channel_3d").set("zmax", "20.500001[m]");
  }

  private static void createChannelMaterial(Model model) {
    model.component("comp1").material().create("mat_channel_3d", "Common");
    model.component("comp1").material("mat_channel_3d").selection().named("sel_channel_3d");
    model.component("comp1").material("mat_channel_3d").propertyGroup("def").set(
        "electricconductivity",
        new String[] {"sigma_channel", "0", "0", "0", "sigma_channel", "0", "0", "0", "sigma_channel"});
    model.component("comp1").material("mat_channel_3d").propertyGroup("def").set(
        "relpermittivity",
        new String[] {"eps_r", "0", "0", "0", "eps_r", "0", "0", "0", "eps_r"});
    model.component("comp1").material("mat_channel_3d").propertyGroup("def").set(
        "relpermeability",
        new String[] {"mu_r", "0", "0", "0", "mu_r", "0", "0", "0", "mu_r"});
  }

  private static void createChannelMefFeature(Model model) {
    model.component("comp1").physics("mef").create("al_channel_3d", "ElectromagneticModel", 3);
    model.component("comp1").physics("mef").feature("al_channel_3d")
        .selection().named("sel_channel_3d");
    model.component("comp1").physics("mef").feature("al_channel_3d").set("mur_mat", "userdef");
    model.component("comp1").physics("mef").feature("al_channel_3d").set("mur", "mu_r");
    model.component("comp1").physics("mef").feature("al_channel_3d").set("sigma_mat", "userdef");
    model.component("comp1").physics("mef").feature("al_channel_3d").set("sigma", "sigma_channel");
    model.component("comp1").physics("mef").feature("al_channel_3d").set("epsilonr_mat", "userdef");
    model.component("comp1").physics("mef").feature("al_channel_3d").set("epsilonr", "eps_r");
    model.component("comp1").physics("mef").feature("al_channel_3d").set("materialType", "nonSolid");
  }

  private static void createChannelMeshControl(Model model) {
    model.component("comp1").mesh("mesh1").feature().create("size_channel_3d", "Size");
    // The canonical mesh sequence ends with ftet1.  Size controls appended
    // after that node are never consumed by the free-tetrahedral operation.
    model.component("comp1").mesh("mesh1").feature().move("size_channel_3d", 4);
    model.component("comp1").mesh("mesh1").feature("size_channel_3d")
        .selection().named("sel_channel_3d");
    model.component("comp1").mesh("mesh1").feature("size_channel_3d").set("custom", "on");
    model.component("comp1").mesh("mesh1").feature("size_channel_3d").set("hmax", "0.25");
    model.component("comp1").mesh("mesh1").feature("size_channel_3d").set("hmin", "0.125");
    model.component("comp1").mesh("mesh1").feature("size_channel_3d").set("hgrad", "1.2");
  }

  private static void configureReceivers(Model model, String outputCsv) {
    model.result().dataset("recvpts").set("data", "dset1");
    model.result().dataset("recvpts").set("pointx", new double[] {0, 0, 0, 0, 0});
    model.result().dataset("recvpts").set("pointy", new double[] {-20, -10, 0, 10, 20});
    model.result().dataset("recvpts").set("pointz", new double[] {-0.1, -0.1, -0.1, -0.1, -0.1});
    String[] expressions = new String[] {"mef.Ex", "d(mef.Bz,t)", "mef.Bz/mu0_const"};
    model.result().export("recv_csv").set("data", "recvpts");
    model.result().export("recv_csv").set("expr", expressions);
    model.result().export("recv_csv").set("descr", new String[] {"Ex", "dBzdt", "Hz"});
    model.result().export("recv_csv").set("filename", outputCsv);
  }

  private static void configureSolver(Model model) {
    model.study("std1").feature("time").set("tlist", TLIST);
    model.study("std1").feature("time").set("rtol", "1e-4");
    model.sol("sol1").feature("t1").set("tlist", TLIST);
    model.sol("sol1").feature("t1").set("rtol", "1e-4");
    model.sol("sol1").feature("t1").set("consistent", "bweuler");
    model.sol("sol1").feature("t1").set("initialstepbdf", "1e-8");
    model.sol("sol1").feature("t1").set("initialstepbdfactive", "on");
    model.sol("sol1").feature("t1").set("bwinitstepfrac", "1");
    model.sol("sol1").feature("t1").set("maxstepconstraintbdf", "expr");
    model.sol("sol1").feature("t1").set("maxstepexpressionbdf",
        "max(1e-8[s],min(2.5e-5[s],0.01*(t+1e-6[s])))");
    model.sol("sol1").feature("s1").feature("dDef").set("ooc", "on");
    model.sol("sol1").feature("t1").feature("dDef").set("ooc", "on");
  }

  private static void setStationaryConfiguration(Model model, boolean channelEnabled) {
    model.component("comp1").physics("ec").active(true);
    setMefFeatureActive(model, "src_earth", true);
    setMefFeatureActive(model, "edge_src_stat", true);
    setMefFeatureActive(model, "al_air", true);
    setMefFeatureActive(model, "al_earth", true);
    setMefFeatureActive(model, "gauge_stat", true);
    if (channelEnabled) setMefFeatureActive(model, "al_channel_3d", true);
  }

  private static void setTransientConfiguration(Model model, boolean channelEnabled) {
    model.component("comp1").physics("ec").active(false);
    setMefFeatureActive(model, "src_earth", false);
    setMefFeatureActive(model, "edge_src_stat", false);
    setMefFeatureActive(model, "al_air", true);
    setMefFeatureActive(model, "al_earth", true);
    setMefFeatureActive(model, "gauge_stat", true);
    if (channelEnabled) setMefFeatureActive(model, "al_channel_3d", true);
  }

  private static void setMefFeatureActive(Model model, String tag, boolean active) {
    model.component("comp1").physics("mef").feature(tag).active(active);
  }

  private static void removeChannelFeatures(Model model) {
    try { model.component("comp1").mesh("mesh1").feature().remove("size_channel_3d"); }
    catch (Exception ignored) {}
    try { model.component("comp1").physics("mef").feature().remove("al_channel_3d"); }
    catch (Exception ignored) {}
    try { model.component("comp1").material().remove("mat_channel_3d"); }
    catch (Exception ignored) {}
    try { model.component("comp1").selection().remove("sel_channel_3d"); }
    catch (Exception ignored) {}
    try { model.component("comp1").geom("geom1").feature().remove("blk_channel_3d"); }
    catch (Exception ignored) {}
  }
}
