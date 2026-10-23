// Copyright (c) 2019 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
//
// SPDX-License-Identifier: GPL-2.0-only

/**
 * @file cttc-nr-demo.cc
 * @brief Smart-campus antenna control scenario on the e-Um 5G band (4.77 GHz).
 *
 * Started from the 5G-LENA cttc-nr-demo example and keeps its program name, which the Python
 * scripts use to run it. Both modes replay a campus day from per-UE waypoints over the real
 * sites, one simulated second per minute, through the system-level CoverageModel (3GPP
 * TR 38.901 UMi pathloss; no NR protocol stack, no packets):
 *
 * - dataset mode (--dataset): antennas fixed; writes the per-cell, per-minute observations of
 *   the TCN training dataset (DatasetCollector).
 * - RL mode (--rlEnv): a Python agent sets the antenna count of every cell each minute through
 *   the ns3-ai Gym interface (RlAntennaEnv).
 */

#include "ns3/ai-module.h"
#include "ns3/core-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/propagation-module.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <vector>

using namespace ns3;

namespace
{

/**
 * System-level coverage model. Decides which cell serves a given point using
 * the ns-3 3GPP UMi propagation loss (deterministic: LOS, no shadowing). Each
 * active antenna transmits at a fixed PA output, so a gNB's total power scales
 * with its active antennas, and the array gain 10*log10(numAntennas) applies
 * only toward the UEs it serves; toward other cells' UEs a gNB is seen without
 * array gain. A cell with more antennas therefore serves points farther away
 * -> larger coverage. This makes antenna count drive coverage without a full
 * packet/REM simulation, while still using the real ns-3 propagation model.
 * Optionally, UEs inside a building see a fixed building entry loss on every
 * link (SetBuildingEntryLoss).
 */
class CoverageModel
{
  public:
    /**
     * Constructor.
     *
     * @param freqHz Carrier frequency (Hz).
     * @param gnbNodes gNB nodes (each must have a mobility model).
     * @param cellOfGnb Map from gNB index to stable cell id (1..N).
     * @param antennasByCell Antenna count per cell id (index 1..N).
     */
    CoverageModel(double freqHz,
                  const NodeContainer& gnbNodes,
                  const std::vector<uint32_t>& cellOfGnb,
                  const std::vector<uint32_t>& antennasByCell)
    {
        m_loss = CreateObject<ThreeGppUmiStreetCanyonPropagationLossModel>();
        m_loss->SetFrequency(freqHz);
        m_loss->SetAttribute("ShadowingEnabled", BooleanValue(false));
        m_loss->SetChannelConditionModel(CreateObject<AlwaysLosChannelConditionModel>());
        m_txDbm.resize(gnbNodes.GetN());
        m_bfGainDb.resize(gnbNodes.GetN());
        for (uint32_t g = 0; g < gnbNodes.GetN(); ++g)
        {
            m_gnbMob.push_back(gnbNodes.Get(g)->GetObject<MobilityModel>());
            uint32_t cell = cellOfGnb[g];
            uint32_t n = (cell < antennasByCell.size()) ? antennasByCell[cell] : 32;
            SetLink(g, n);
            m_radiusM.push_back(CoverageRadiusM(n));
            m_cellOfGnb.push_back(cell);
        }
        // Thermal noise floor for the wideband SINR: -174 dBm/Hz + 10log10(BW) + NF.
        const double bwHz = 100e6; // nominal NR FR1 carrier bandwidth
        const double nfDb = 9.0;   // UE receiver noise figure
        double noiseDbm = -174.0 + 10.0 * log10(bwHz) + nfDb;
        m_noiseW = std::pow(10.0, (noiseDbm - 30.0) / 10.0);
    }

    /**
     * Enable the building entry loss of indoor UEs, as one fixed value (e.g. the ITU-R
     * P.2109 median). A UE is indoors when it is within @p radiusM (horizontal) of an
     * indoor site, i.e. in the area where the scenario places a building's occupants.
     * The loss applies to every link of that UE alike, so it lowers the UE's SINR only
     * through the noise term and never changes which cell serves it.
     *
     * @param indoorSites Positions of the sites that are buildings (not open spaces).
     * @param radiusM Horizontal distance from an indoor site counted as inside it (m).
     * @param lossDb Building entry loss (dB); 0 disables it.
     */
    void SetBuildingEntryLoss(const std::vector<Vector>& indoorSites, double radiusM, double lossDb)
    {
        m_indoorSites = indoorSites;
        m_indoorRadiusM = radiusM;
        m_entryLossDb = lossDb;
    }

    /**
     * Serving cell plus wideband SINR (dB) at a UE position: beamformed power from
     * the best cell over the summed (non-beamformed) power of all other cells
     * plus thermal noise.
     *
     * @param ueMob The UE mobility model.
     * @param sinrDbOut Set to the wideband SINR (dB).
     * @return the stable cell id (1..N) serving the position.
     */
    uint32_t ServingCellSinr(Ptr<MobilityModel> ueMob, double& sinrDbOut) const
    {
        std::vector<double> rxDbm(m_gnbMob.size());
        double best = -std::numeric_limits<double>::max();
        uint32_t bestG = 0;
        const double lossDb = EntryLossDb(ueMob);
        for (uint32_t g = 0; g < m_gnbMob.size(); ++g)
        {
            rxDbm[g] = RxDbm(g, ueMob, lossDb);
            if (rxDbm[g] + m_bfGainDb[g] > best)
            {
                best = rxDbm[g] + m_bfGainDb[g];
                bestG = g;
            }
        }
        double sigW = std::pow(10.0, (best - 30.0) / 10.0);
        double intfW = 0.0;
        for (uint32_t g = 0; g < rxDbm.size(); ++g)
        {
            if (g != bestG)
            {
                intfW += std::pow(10.0, (rxDbm[g] - 30.0) / 10.0);
            }
        }
        sinrDbOut = 10.0 * std::log10(sigW / (intfW + m_noiseW));
        return m_cellOfGnb.empty() ? 0 : m_cellOfGnb[bestG];
    }

    /**
     * Nearest cell to a position within a sensing radius, by geometry only
     * (independent of connection / best-server). Used for ISAC presence density:
     * a device is "sensed" by the closest cell if within @p radius, whether or
     * not it is network-connected (or on campus).
     *
     * @param ueMob The UE mobility model.
     * @param radius Maximum sensing distance (m).
     * @param distOut Set to the distance (m) to the nearest gNB.
     * @return the nearest cell id (1..N), or 0 if none is within @p radius.
     */
    uint32_t NearestCellWithin(Ptr<MobilityModel> ueMob, double radius, double& distOut) const
    {
        double best = std::numeric_limits<double>::max();
        uint32_t bestCell = 0;
        for (uint32_t g = 0; g < m_gnbMob.size(); ++g)
        {
            double dd = CalculateDistance(ueMob->GetPosition(), m_gnbMob[g]->GetPosition());
            if (dd < best)
            {
                best = dd;
                bestCell = m_cellOfGnb[g];
            }
        }
        distOut = best;
        return (best <= radius) ? bestCell : 0;
    }

    /**
     * Replace the per-cell antenna counts at run time.
     *
     * The antenna count sets a gNB's total transmit power and its beamforming
     * gain, so the recomputed values change best-server association, SINR and
     * therefore handovers from the next evaluation onwards.
     *
     * @param antennasByCell Antenna count per cell id (index 1..N).
     */
    void SetAntennas(const std::vector<uint32_t>& antennasByCell)
    {
        for (uint32_t g = 0; g < m_cellOfGnb.size(); ++g)
        {
            uint32_t cell = m_cellOfGnb[g];
            uint32_t n = (cell < antennasByCell.size()) ? antennasByCell[cell] : 16;
            SetLink(g, n);
            m_radiusM[g] = CoverageRadiusM(n);
        }
    }

    /**
     * Effective coverage radius for an active antenna count (2/4/8/16 -> 40/55/70/85 m).
     *
     * @param n Active antenna count.
     * @return the horizontal coverage radius (m).
     */
    static double CoverageRadiusM(uint32_t n)
    {
        if (n <= 2)
        {
            return 40.0;
        }
        if (n <= 4)
        {
            return 55.0;
        }
        if (n <= 8)
        {
            return 70.0;
        }
        return 85.0;
    }

    /**
     * Serving cell under the coverage-radius rule: a UE attaches to the strongest
     * cell among those whose radius covers it horizontally, so shrinking a cell
     * pushes its edge UEs to a neighbor that still reaches them. If no cell covers
     * the UE it is out of coverage; it is then reported against the strongest cell
     * overall, so the cell that shrank carries the unserved demand.
     *
     * @param ueMob The UE mobility model.
     * @param sinrDbOut Set to the wideband SINR (dB) toward the returned cell.
     * @param coveredOut Set to whether any cell covers the UE.
     * @param inServiceAreaOut Set to whether the UE would be covered with every cell
     *        at its largest radius, i.e. whether an outage is attributable to control.
     * @return the stable cell id (1..N) the UE is attributed to.
     */
    uint32_t ServingCellSinrInCoverage(Ptr<MobilityModel> ueMob,
                                       double& sinrDbOut,
                                       bool& coveredOut,
                                       bool& inServiceAreaOut) const
    {
        const Vector u = ueMob->GetPosition();
        std::vector<double> rxDbm(m_gnbMob.size());
        double bestAll = -std::numeric_limits<double>::max();
        double bestCov = -std::numeric_limits<double>::max();
        uint32_t gAll = 0;
        int gCov = -1;
        const double maxRadius = CoverageRadiusM(std::numeric_limits<uint32_t>::max());
        const double lossDb = EntryLossDb(ueMob);
        inServiceAreaOut = false;
        for (uint32_t g = 0; g < m_gnbMob.size(); ++g)
        {
            rxDbm[g] = RxDbm(g, ueMob, lossDb);
            const double servedDbm = rxDbm[g] + m_bfGainDb[g];
            if (servedDbm > bestAll)
            {
                bestAll = servedDbm;
                gAll = g;
            }
            const Vector b = m_gnbMob[g]->GetPosition();
            const double dh = std::hypot(u.x - b.x, u.y - b.y);
            inServiceAreaOut = inServiceAreaOut || (dh <= maxRadius);
            if (dh <= m_radiusM[g] && servedDbm > bestCov)
            {
                bestCov = servedDbm;
                gCov = static_cast<int>(g);
            }
        }
        coveredOut = (gCov >= 0);
        const uint32_t serving = coveredOut ? static_cast<uint32_t>(gCov) : gAll;
        double sigW = std::pow(10.0, (rxDbm[serving] + m_bfGainDb[serving] - 30.0) / 10.0);
        double intfW = 0.0;
        for (uint32_t g = 0; g < rxDbm.size(); ++g)
        {
            if (g != serving)
            {
                intfW += std::pow(10.0, (rxDbm[g] - 30.0) / 10.0);
            }
        }
        sinrDbOut = 10.0 * std::log10(sigW / (intfW + m_noiseW));
        return m_cellOfGnb.empty() ? 0 : m_cellOfGnb[serving];
    }

  private:
    /**
     * Set a gNB's link parameters for @p n active antennas. Each active antenna
     * keeps its PA output (TR 38.864 spatial-domain adaptation), so the total
     * transmit power scales with n, while the array gain 10*log10(n) is gained
     * only toward the UEs the gNB serves.
     *
     * @param g The gNB index.
     * @param n The active antenna count.
     */
    void SetLink(uint32_t g, uint32_t n)
    {
        const double a = static_cast<double>(std::max<uint32_t>(1, n));
        m_txDbm[g] = m_txPowerDbm + 10.0 * log10(a / kRefAntennas);
        m_bfGainDb[g] = 10.0 * log10(a);
    }

    /**
     * Received power from a gNB without its beamforming gain. Averaged over beam
     * directions an array radiates no gain, so this is the interference level a
     * gNB causes at a UE it does not serve.
     *
     * @param g The gNB index.
     * @param ueMob The UE mobility model.
     * @param entryLossDb The UE's building entry loss (EntryLossDb).
     * @return the received power (dBm).
     */
    double RxDbm(uint32_t g, Ptr<MobilityModel> ueMob, double entryLossDb) const
    {
        return m_loss->CalcRxPower(m_txDbm[g], m_gnbMob[g], ueMob) - entryLossDb;
    }

    /**
     * @param ueMob The UE mobility model.
     * @return the building entry loss (dB) of the UE: the configured loss when it is inside
     *         an indoor site's radius, else 0.
     */
    double EntryLossDb(Ptr<MobilityModel> ueMob) const
    {
        if (m_entryLossDb <= 0.0)
        {
            return 0.0;
        }
        constexpr double kRoundingM = 0.01; // waypoint coordinates are written to 1 cm
        const Vector u = ueMob->GetPosition();
        for (const auto& s : m_indoorSites)
        {
            if (std::hypot(u.x - s.x, u.y - s.y) <= m_indoorRadiusM + kRoundingM)
            {
                return m_entryLossDb;
            }
        }
        return 0.0;
    }

    static constexpr double kRefAntennas = 16.0; ///< antenna count at which m_txPowerDbm applies

    Ptr<ThreeGppUmiStreetCanyonPropagationLossModel> m_loss; ///< 3GPP UMi pathloss
    std::vector<Ptr<MobilityModel>> m_gnbMob;                ///< gNB mobility models
    std::vector<double> m_txDbm;                             ///< per-gNB total tx power (dBm)
    std::vector<double> m_bfGainDb;                          ///< per-gNB beamforming gain (dB)
    std::vector<double> m_radiusM;                           ///< per-gNB coverage radius (m)
    std::vector<uint32_t> m_cellOfGnb;                       ///< gNB index -> cell id
    double m_txPowerDbm{30.0};         ///< per-gNB total tx power with all antennas on (dBm)
    double m_noiseW{0.0};              ///< thermal noise power (W)
    std::vector<Vector> m_indoorSites; ///< positions of the sites that are buildings
    double m_indoorRadiusM{0.0};       ///< distance from an indoor site counted as inside (m)
    double m_entryLossDb{0.0};         ///< building entry loss of indoor UEs (dB), 0 = none
};

/**
 * All tunable parameters of the simulation, plus command-line parsing.
 */
struct SimConfig
{
    // Per-cell antennas: coverage depends on antenna count (more antennas ->
    // larger coverage). Default uniform gnbAntennas; a CSV (cell,numAntennas)
    // overrides per cell. Applies to the dataset mode (RL starts from rlAntennaInit).
    uint32_t gnbAntennas = 16;   // default active antennas per cell (all on)
    std::string antennaFile = ""; // optional CSV: cell,numAntennas

    // Real site (building) placement: CSV with x,y columns. Each site becomes
    // one gNB = one cell = one control unit (building).
    std::string sitesFile = "";

    // Per-UE waypoints (CSV: ueId,t,x,y[,on]) for scenario-driven movement
    // (UEs move between buildings / off-campus per the scenario, interpolated).
    std::string waypoints = "";

    // Dataset mode: step the scenario minute-by-minute (waypoint UEs) and record
    // per-cell RF/geometry observations (ue_count, isac_count, mean_sinr_db,
    // handover_in/out) for the TCN training dataset. Requires --waypoints (in real
    // coordinates, t in minutes) and --sitesFile. Antennas fixed (baseline).
    bool dataset = false;
    uint32_t datasetMinutes = 720; // 09:00-21:00 at 1-min resolution
    // ISAC sensing radius (m): a device is sensed by the nearest cell within this,
    // regardless of connection status (presence-based Isac_density).
    double isacSenseRadius = 100.0;
    // Per-UE stochastic demand, connection delay (presence vs connected split) and P_D
    // device-free sensing; the "v2" names come from dataset generation version 2.
    std::string ratesFile = "";     // CSV (cell,baseRate,eveningMult[,lagMin]) per-cell demand
    double v2ConnDelayMean = 3.0;   // mean per-UE connection delay (minutes)
    double v2ActivityProb = 0.7;    // per-UE per-minute traffic activity probability
    double v2RateCv = 0.3;          // per-UE demand multiplier spread (uniform +/-)
    double v2Pd = 0.9;              // ISAC detection probability per present target

    // RL antenna-control environment (ns3-ai Gym interface). Uses the same
    // scenario and observation semantics as the dataset mode.
    bool rlEnv = false;             // RL: run as a Gym environment instead of writing a CSV
    uint32_t rlAntennaInit = 16;    // RL: antenna count per cell before the first action
    double rlSinrThresholdDb = -5.0; // RL: connected UEs below this SINR count as dropped
    uint32_t rlOutageSamplesPerMin = 1; // RL: outage samples per minute (1 = the step only)
    double rlOutageEventSec = 10.0; // RL: an outage run at least this long is a long event (s)
    std::string rlShmTag;           // RL: suffix of the ns3-ai shared-memory names

    // Radio parameters of the system-level coverage model. Defaults model the Korean
    // private 5G band (e-Um 5G): 4.72-4.82 GHz, inside 3GPP band n79.
    double slmCarrierFrequency = 4.77e9; // carrier (Hz): center of the 4.72-4.82 GHz block
    // Building entry loss of indoor UEs (dB) in the dataset and RL modes: ITU-R P.2109-2
    // median for a traditional building at 4.77 GHz.
    double o2iLossDb = 16.2;
    double o2iIndoorRadius = 18.0; // a UE this close to an indoor site is inside it (m)

    // Where we will store the output files.
    std::string simTag = "default";
    std::string outputDir = "./";

    /**
     * Register every parameter with the ns-3 CommandLine, parse argv and check the
     * radio parameters.
     *
     * @param argc The argument count from main.
     * @param argv The argument vector from main.
     */
    void Parse(int argc, char* argv[]);
};

void
SimConfig::Parse(int argc, char* argv[])
{
    CommandLine cmd(__FILE__);

    cmd.AddValue("gnbAntennas", "Default active antennas of a cell not listed in antennaFile", gnbAntennas);
    cmd.AddValue("antennaFile", "CSV (cell,numAntennas) for per-cell antenna counts", antennaFile);
    cmd.AddValue("sitesFile", "CSV with x,y columns: place one gNB per site (real deployment)",
                 sitesFile);
    cmd.AddValue("waypoints",
                 "CSV (ueId,t,x,y): per-UE waypoints for scenario-driven movement",
                 waypoints);
    cmd.AddValue("dataset",
                 "Dataset mode: step the scenario minute-by-minute and record per-cell"
                 " RF observations (ue_count, isac_count, mean_sinr_db, handover) for TCN",
                 dataset);
    cmd.AddValue("datasetMinutes", "Number of 1-minute steps to record in dataset mode",
                 datasetMinutes);
    cmd.AddValue("isacSenseRadius",
                 "ISAC sensing radius (m): a device counts for the nearest cell within this,"
                 " independent of connection (presence-based Isac_density)",
                 isacSenseRadius);
    cmd.AddValue("ratesFile", "CSV (cell,baseRate,eveningMult[,lagMin]) per-cell demand",
                 ratesFile);
    cmd.AddValue("v2ConnDelayMean", "Mean per-UE connection delay (min)", v2ConnDelayMean);
    cmd.AddValue("v2ActivityProb", "Per-UE per-minute traffic activity probability",
                 v2ActivityProb);
    cmd.AddValue("v2RateCv", "Per-UE demand multiplier spread (uniform +/-)", v2RateCv);
    cmd.AddValue("v2Pd", "ISAC detection probability per present target", v2Pd);
    cmd.AddValue("rlEnv",
                 "RL: expose the scenario as an ns3-ai Gym environment for antenna control",
                 rlEnv);
    cmd.AddValue("rlAntennaInit", "RL: antenna count per cell before the first action",
                 rlAntennaInit);
    cmd.AddValue("rlSinrThresholdDb",
                 "RL: connected UEs below this SINR (dB) are counted as dropped",
                 rlSinrThresholdDb);
    cmd.AddValue("rlOutageSamplesPerMin",
                 "RL: outage samples per simulated minute for the campus outage counters in "
                 "the extra info (1 = the minute step only, 60 = every second); the "
                 "observation and rewards do not depend on it",
                 rlOutageSamplesPerMin);
    cmd.AddValue("rlOutageEventSec",
                 "RL: an outage run of at least this many seconds counts as a long outage event",
                 rlOutageEventSec);
    cmd.AddValue("rlShmTag",
                 "RL: suffix of the ns3-ai shared-memory names, set by the Python side so "
                 "several RL processes can run on one machine (empty: ns3-ai defaults)",
                 rlShmTag);
    cmd.AddValue("slmCarrierFrequency",
                 "Carrier frequency (Hz) of the system-level coverage model",
                 slmCarrierFrequency);
    cmd.AddValue("o2iLossDb",
                 "Building entry loss (dB) of indoor UEs in the dataset and RL modes "
                 "(0 = none). A UE is indoors within o2iIndoorRadius of a site whose "
                 "'indoor' column in the sites file is 1 (or absent)",
                 o2iLossDb);
    cmd.AddValue("o2iIndoorRadius",
                 "Horizontal distance (m) from an indoor site within which a UE is inside it",
                 o2iIndoorRadius);
    cmd.AddValue("simTag",
                 "tag to be appended to output filenames to distinguish simulation campaigns",
                 simTag);
    cmd.AddValue("outputDir", "directory where to store simulation results", outputDir);

    cmd.Parse(argc, argv);

    NS_ABORT_MSG_IF(slmCarrierFrequency < 0.5e9 || slmCarrierFrequency > 100e9,
                    "slmCarrierFrequency must be within 0.5-100 GHz (TR 38.901)");
    NS_ABORT_MSG_IF(o2iLossDb < 0.0 || o2iIndoorRadius < 0.0,
                    "o2iLossDb and o2iIndoorRadius must not be negative");
}

/**
 * Load per-cell antenna counts from a CSV (columns: cell,numAntennas), indexed
 * by stable cell id (1..nCells). Cells not listed use the uniform default.
 *
 * @param cfg The simulation configuration.
 * @param nCells Number of cells.
 * @return vector of size nCells+1 (index 1..nCells) with antenna counts.
 */
std::vector<uint32_t>
LoadAntennas(const SimConfig& cfg, uint32_t nCells)
{
    std::vector<uint32_t> a(nCells + 1, cfg.gnbAntennas);
    if (cfg.antennaFile.empty())
    {
        return a;
    }
    std::ifstream f(cfg.antennaFile.c_str());
    if (!f.is_open())
    {
        std::cerr << "Can't open antenna file " << cfg.antennaFile << std::endl;
        return a;
    }
    std::string line;
    std::getline(f, line); // header
    while (std::getline(f, line))
    {
        std::stringstream ss(line);
        std::string cellStr;
        std::string antStr;
        std::getline(ss, cellStr, ',');
        std::getline(ss, antStr, ',');
        if (cellStr.empty() || antStr.empty())
        {
            continue;
        }
        uint32_t c = static_cast<uint32_t>(std::stoul(cellStr));
        uint32_t n = static_cast<uint32_t>(std::stoul(antStr));
        if (c >= 1 && c <= nCells)
        {
            a[c] = n;
        }
    }
    return a;
}

/** Per-cell demand parameters of the dataset and RL modes. */
struct CellRate
{
    double baseRate{0.5};    ///< per-person demand (Mbps)
    double eveningMult{1.0}; ///< demand multiplier after 18:00
    double lagMin{0.0};      ///< lead of arrival over first traffic (min, per building type)
};

/**
 * Load per-cell demand parameters from a CSV (columns: cell,baseRate,eveningMult[,lagMin]),
 * indexed by stable cell id (1..nCells). lagMin is the building-type lead of people over
 * traffic: a UE arriving at the cell waits this long plus its own connection delay before
 * it connects. Cells not listed, or files without lagMin, use (0.5, 1.0, 0).
 *
 * @param path CSV path.
 * @param nCells Number of cells.
 * @return vector of size nCells+1 (index 1..nCells).
 */
std::vector<CellRate>
LoadRates(const std::string& path, uint32_t nCells)
{
    std::vector<CellRate> r(nCells + 1);
    std::ifstream f(path.c_str());
    NS_ABORT_MSG_IF(!f.is_open(), "Can't open rates file " + path);
    std::string line;
    std::getline(f, line); // header
    while (std::getline(f, line))
    {
        std::stringstream ss(line);
        std::string cellStr;
        std::string rateStr;
        std::string eveStr;
        std::string lagStr;
        std::getline(ss, cellStr, ',');
        std::getline(ss, rateStr, ',');
        std::getline(ss, eveStr, ',');
        std::getline(ss, lagStr, ',');
        if (cellStr.empty() || rateStr.empty())
        {
            continue;
        }
        uint32_t c = static_cast<uint32_t>(std::stoul(cellStr));
        if (c >= 1 && c <= nCells)
        {
            r[c].baseRate = std::stod(rateStr);
            r[c].eveningMult = eveStr.empty() ? 1.0 : std::stod(eveStr);
            r[c].lagMin = lagStr.empty() ? 0.0 : std::stod(lagStr);
        }
    }
    return r;
}

/**
 * Load site (building) positions from a CSV with 'x' and 'y' columns (found by
 * header name). Each site becomes one gNB = one cell = one control unit.
 *
 * @param path CSV path.
 * @param z The height (z) to assign to each site.
 * @param indoorOut If not null, set to whether each site is a building (optional 'indoor'
 *        column: 0 = open space such as a plaza or a sports field; absent or empty = 1).
 * @return the site positions.
 */
std::vector<Vector>
LoadSitePositions(const std::string& path, double z, std::vector<bool>* indoorOut = nullptr)
{
    std::vector<Vector> pos;
    std::vector<bool> indoor;
    std::ifstream f(path.c_str());
    NS_ABORT_MSG_IF(!f.is_open(), "Can't open sites file " + path);
    std::string header;
    std::getline(f, header);
    int ix = -1;
    int iy = -1;
    int iIndoor = -1;
    std::stringstream hs(header);
    std::string col;
    int idx = 0;
    while (std::getline(hs, col, ','))
    {
        if (col == "x")
        {
            ix = idx;
        }
        else if (col == "y")
        {
            iy = idx;
        }
        else if (col == "indoor")
        {
            iIndoor = idx;
        }
        ++idx;
    }
    NS_ABORT_MSG_IF(ix < 0 || iy < 0, "sites file needs 'x' and 'y' columns");
    std::string line;
    while (std::getline(f, line))
    {
        if (line.empty())
        {
            continue;
        }
        std::vector<std::string> cols;
        std::stringstream ls(line);
        std::string c;
        while (std::getline(ls, c, ','))
        {
            cols.push_back(c);
        }
        if (static_cast<int>(cols.size()) <= std::max(ix, iy))
        {
            continue;
        }
        pos.push_back(Vector(std::stod(cols[ix]), std::stod(cols[iy]), z));
        const bool hasFlag =
            iIndoor >= 0 && iIndoor < static_cast<int>(cols.size()) && !cols[iIndoor].empty();
        indoor.push_back(!hasFlag || std::stoi(cols[iIndoor]) != 0);
    }
    if (indoorOut)
    {
        *indoorOut = indoor;
    }
    return pos;
}

/**
 * @param sitePos Site positions (LoadSitePositions).
 * @param indoor Whether each site is a building (LoadSitePositions).
 * @return the positions of the sites that are buildings.
 */
std::vector<Vector>
IndoorSitePositions(const std::vector<Vector>& sitePos, const std::vector<bool>& indoor)
{
    std::vector<Vector> out;
    for (uint32_t i = 0; i < sitePos.size(); ++i)
    {
        if (indoor[i])
        {
            out.push_back(sitePos[i]);
        }
    }
    return out;
}

/**
 * Load per-UE waypoints (CSV: ueId,t,x,y,on) and create one UE node per id with a
 * WaypointMobilityModel driven by its (sorted) waypoints. Coordinates are used
 * as-is; @p t is in the simulation's time unit. The optional @p on column marks
 * whether the UE is on campus at that waypoint (default true).
 *
 * @param path Waypoints CSV path.
 * @param ues [out] Created UE nodes with a WaypointMobilityModel installed.
 * @param onByUe [out] Per-UE sorted (time, on-campus?) schedule (indexed by id).
 * @return Number of UEs created (max id + 1).
 */
uint32_t
LoadWaypointUes(const std::string& path,
                NodeContainer& ues,
                std::vector<std::vector<std::pair<double, bool>>>& onByUe)
{
    std::map<uint32_t, std::vector<std::pair<double, Vector>>> wpByUe;
    std::map<uint32_t, std::vector<std::pair<double, bool>>> onMap;
    std::ifstream wf(path.c_str());
    NS_ABORT_MSG_IF(!wf.is_open(), "Cannot open waypoints file: " + path);
    std::string line;
    std::getline(wf, line); // header: ueId,t,x,y[,on]
    uint32_t maxUe = 0;
    while (std::getline(wf, line))
    {
        if (line.empty())
        {
            continue;
        }
        std::stringstream ss(line);
        std::string idS;
        std::string tS;
        std::string xS;
        std::string yS;
        std::string onS;
        std::getline(ss, idS, ',');
        std::getline(ss, tS, ',');
        std::getline(ss, xS, ',');
        std::getline(ss, yS, ',');
        std::getline(ss, onS, ',');
        if (yS.empty())
        {
            continue;
        }
        uint32_t id = static_cast<uint32_t>(std::stoul(idS));
        double t = std::stod(tS);
        wpByUe[id].push_back({t, Vector(std::stod(xS), std::stod(yS), 1.5)});
        onMap[id].push_back({t, onS.empty() || std::stoi(onS) != 0});
        maxUe = std::max(maxUe, id);
    }
    const uint32_t nUe = wpByUe.empty() ? 0 : maxUe + 1;
    ues.Create(nUe);
    MobilityHelper mob;
    mob.SetMobilityModel("ns3::WaypointMobilityModel");
    mob.Install(ues);
    onByUe.assign(nUe, {});
    for (uint32_t i = 0; i < nUe; ++i)
    {
        Ptr<WaypointMobilityModel> wm = ues.Get(i)->GetObject<WaypointMobilityModel>();
        auto it = wpByUe.find(i);
        if (it == wpByUe.end())
        {
            continue;
        }
        std::sort(it->second.begin(), it->second.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        for (const auto& w : it->second)
        {
            wm->AddWaypoint(Waypoint(Seconds(w.first), w.second));
        }
        auto on = onMap[i];
        std::sort(on.begin(), on.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        onByUe[i] = std::move(on);
    }
    return nUe;
}

/**
 * Dataset engine for the TCN training data. Drives the scenario minute-by-minute
 * (waypoint UEs, one sim second = one minute) and records, per cell per minute,
 * the observations that only ns-3 can produce: connected UE count, ISAC detections and
 * people present within the sensing radius, mean SINR (linear-averaged then dB), handover
 * in/out (from serving-cell changes between consecutive minutes) and the demanded traffic
 * of the connected UEs. The remaining dataset features (PRB, neighbor load, ISAC noise,
 * hour) are derived in Python. Antennas are held fixed (the fixed-16 baseline the RL later
 * improves on). Off-campus UEs (on=0) are present but not served/counted.
 */
class DatasetCollector
{
  public:
    /**
     * Constructor.
     *
     * @param cfg The simulation configuration.
     */
    explicit DatasetCollector(const SimConfig& cfg)
        : m_cfg(cfg),
          m_interval(Seconds(1.0))
    {
    }

    /** Build the scenario, step minute-by-minute and write <simTag>-obs.csv. */
    void Run();

  private:
    /** Record one minute of per-cell observations and reschedule. */
    void Sample();

    /** Whether UE @p i is on campus at the current sim time (step function). */
    bool OnNow(uint32_t i) const;

    const SimConfig& m_cfg;                     ///< Simulation configuration
    Time m_interval;                            ///< Sampling period (1 min)
    NodeContainer m_gnbs;                       ///< gNB nodes
    NodeContainer m_ues;                        ///< Waypoint UE nodes
    std::vector<uint32_t> m_cellOfGnb;          ///< gNB index -> cell id (1..N)
    std::vector<Vector> m_cellPos;              ///< cell id -> gNB position (1..N)
    std::vector<uint32_t> m_antennas;           ///< cell id -> antenna count (1..N)
    std::vector<std::vector<std::pair<double, bool>>> m_ueOn; ///< per-UE on-campus schedule
    std::vector<uint32_t> m_prevCell;           ///< per-UE serving cell last minute (0=none)
    std::unique_ptr<CoverageModel> m_coverage;  ///< Antenna-aware coverage / SINR
    std::ofstream m_out;                        ///< Output CSV stream
    uint32_t m_nCells{0};                       ///< Number of cells
    bool m_first{true};                         ///< First sample (skip handover counting)

    // Per-UE stochastic state. Times are sim seconds (1 s = 1 min).
    std::vector<double> m_rateMult;             ///< per-UE demand multiplier
    std::vector<double> m_connDelay;            ///< per-UE connection delay (min)
    std::vector<double> m_ueArm;                ///< per-UE last arrival + building lead (min)
    std::vector<bool> m_prevOn;                 ///< per-UE previous-minute on-campus flag
    std::vector<double> m_baseRate;             ///< cell -> per-person demand (Mbps)
    std::vector<double> m_eveMult;              ///< cell -> evening (>=18:00) multiplier
    std::vector<double> m_lagMin;               ///< cell -> arrival lead over traffic (min)
    Ptr<UniformRandomVariable> m_rng;           ///< RNG for activity / detection
};

bool
DatasetCollector::OnNow(uint32_t i) const
{
    if (i >= m_ueOn.size() || m_ueOn[i].empty())
    {
        return true;
    }
    const double now = Simulator::Now().GetSeconds();
    bool on = m_ueOn[i].front().second;
    for (const auto& e : m_ueOn[i])
    {
        if (e.first <= now)
        {
            on = e.second;
        }
        else
        {
            break;
        }
    }
    return on;
}

void
DatasetCollector::Run()
{
    std::vector<bool> siteIndoor;
    std::vector<Vector> gnbPos = LoadSitePositions(m_cfg.sitesFile, 10.0, &siteIndoor);
    const uint32_t nGnb = gnbPos.size();
    NS_ABORT_MSG_IF(nGnb == 0, "dataset mode needs a --sitesFile");
    NS_ABORT_MSG_IF(m_cfg.waypoints.empty(), "dataset mode needs --waypoints");
    m_nCells = nGnb;

    // Stable cell ids by real (y, then x): the same numbering the RL mode uses.
    std::vector<uint32_t> order(nGnb);
    for (uint32_t i = 0; i < nGnb; ++i)
    {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return (gnbPos[a].y != gnbPos[b].y) ? (gnbPos[a].y < gnbPos[b].y)
                                            : (gnbPos[a].x < gnbPos[b].x);
    });
    m_cellOfGnb.assign(nGnb, 0);
    m_cellPos.assign(nGnb + 1, Vector());
    for (uint32_t rank = 0; rank < nGnb; ++rank)
    {
        m_cellOfGnb[order[rank]] = rank + 1;
        m_cellPos[rank + 1] = gnbPos[order[rank]];
    }

    // gNB nodes at their real positions.
    m_gnbs.Create(nGnb);
    MobilityHelper gnbMob;
    Ptr<ListPositionAllocator> gnbAlloc = CreateObject<ListPositionAllocator>();
    for (const auto& p : gnbPos)
    {
        gnbAlloc->Add(p);
    }
    gnbMob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    gnbMob.SetPositionAllocator(gnbAlloc);
    gnbMob.Install(m_gnbs);

    // Fixed antenna baseline (Python writes all cells = 16) and coverage/SINR.
    m_antennas = LoadAntennas(m_cfg, nGnb);
    m_coverage =
        std::make_unique<CoverageModel>(m_cfg.slmCarrierFrequency, m_gnbs, m_cellOfGnb, m_antennas);
    m_coverage->SetBuildingEntryLoss(IndoorSitePositions(gnbPos, siteIndoor),
                                     m_cfg.o2iIndoorRadius,
                                     m_cfg.o2iLossDb);

    // Waypoint UEs (real coordinates, t in minutes) + per-UE on-campus schedule.
    uint32_t nUe = LoadWaypointUes(m_cfg.waypoints, m_ues, m_ueOn);
    NS_ABORT_MSG_IF(nUe == 0, "dataset mode: no UEs in waypoints file");

    // These draws once picked a fixed ISAC device subset (dataset version 1). They are still
    // made and discarded so that the per-UE attributes below, and with them the published
    // dataset, stay the same for a given RngRun.
    Ptr<UniformRandomVariable> rng = CreateObject<UniformRandomVariable>();
    for (uint32_t i = 0; i < nUe; ++i)
    {
        rng->GetValue();
    }
    m_prevCell.assign(nUe, 0);

    // Per-cell demand rates + per-UE random attributes (demand multiplier,
    // connection delay) and the activity RNG.
    auto rates = LoadRates(m_cfg.ratesFile, nGnb);
    m_baseRate.assign(nGnb + 1, 0.5);
    m_eveMult.assign(nGnb + 1, 1.0);
    m_lagMin.assign(nGnb + 1, 0.0);
    for (uint32_t c = 1; c <= nGnb; ++c)
    {
        m_baseRate[c] = rates[c].baseRate;
        m_eveMult[c] = rates[c].eveningMult;
        m_lagMin[c] = rates[c].lagMin;
    }
    m_rng = CreateObject<UniformRandomVariable>();
    m_rateMult.assign(nUe, 1.0);
    m_connDelay.assign(nUe, 0.0);
    m_ueArm.assign(nUe, -1e12);
    m_prevOn.assign(nUe, false);
    const double lo = std::max(0.1, 1.0 - m_cfg.v2RateCv);
    const double hi = 1.0 + m_cfg.v2RateCv;
    for (uint32_t i = 0; i < nUe; ++i)
    {
        m_rateMult[i] = rng->GetValue(lo, hi); // per-UE demand spread
        // The sim clock runs one second per minute, so minutes are used as-is.
        m_connDelay[i] = rng->GetValue(0.0, 2.0 * m_cfg.v2ConnDelayMean);
    }

    std::string filename = m_cfg.outputDir + "/" + m_cfg.simTag + "-obs.csv";
    m_out.open(filename.c_str(), std::ofstream::out | std::ofstream::trunc);
    NS_ABORT_MSG_IF(!m_out.is_open(), "Can't open dataset file " + filename);
    m_out.setf(std::ios_base::fixed);
    m_out << "minute,cell,gnbX,gnbY,ue_count,isac_count,mean_sinr_db,ho_in,ho_out,"
             "traffic_demand_mbps,presence_count\n";

    Simulator::Schedule(Seconds(0.0), &DatasetCollector::Sample, this);
    Simulator::Stop(Seconds(static_cast<double>(m_cfg.datasetMinutes) + 0.5));
    Simulator::Run();
    Simulator::Destroy();
    m_out.close();
}

void
DatasetCollector::Sample()
{
    const double now = Simulator::Now().GetSeconds();
    const uint32_t minute = static_cast<uint32_t>(std::llround(now));
    const double hour = 9.0 + now / 60.0;

    std::vector<uint32_t> ueCount(m_nCells + 1, 0);
    std::vector<uint32_t> isacCount(m_nCells + 1, 0);
    std::vector<double> sinrLin(m_nCells + 1, 0.0);
    std::vector<uint32_t> hoIn(m_nCells + 1, 0);
    std::vector<uint32_t> hoOut(m_nCells + 1, 0);
    std::vector<double> traffic(m_nCells + 1, 0.0);
    std::vector<uint32_t> presence(m_nCells + 1, 0);

    for (uint32_t i = 0; i < m_ues.GetN(); ++i)
    {
        Ptr<MobilityModel> mob = m_ues.Get(i)->GetObject<MobilityModel>();
        const bool on = OnNow(i);
        uint32_t cur = 0; // serving cell of a *connected* UE (0 = none)

        // Arrival (0->1 on-campus transition) resets the connection-delay clock, which
        // starts after the lead of the building the UE arrives at.
        if (on && !m_prevOn[i])
        {
            double d = 0.0;
            m_ueArm[i] =
                now + m_lagMin[m_coverage->NearestCellWithin(mob,
                                                             std::numeric_limits<double>::max(),
                                                             d)];
        }
        m_prevOn[i] = on;

        // ISAC: device-free presence sensing over ALL present people, each
        // detected with probability P_D within the sensing radius.
        if (on)
        {
            double ds = 0.0;
            uint32_t sc = m_coverage->NearestCellWithin(mob, m_cfg.isacSenseRadius, ds);
            if (sc != 0)
            {
                presence[sc]++;
                if (m_rng->GetValue() < m_cfg.v2Pd)
                {
                    isacCount[sc]++;
                }
            }
        }

        // Connected only after the per-UE connection delay since arrival.
        const bool connected = on && (now >= m_ueArm[i] + m_connDelay[i]);
        if (connected)
        {
            double sinrDb = 0.0;
            cur = m_coverage->ServingCellSinr(mob, sinrDb);
            ueCount[cur]++;
            sinrLin[cur] += std::pow(10.0, sinrDb / 10.0);
            // Per-UE stochastic demand: active this minute w.p. v2ActivityProb.
            // Divide by that prob so the expected level tracks the base rate.
            if (m_rng->GetValue() < m_cfg.v2ActivityProb)
            {
                double eve = (hour >= 18.0) ? m_eveMult[cur] : 1.0;
                traffic[cur] += m_baseRate[cur] * eve * m_rateMult[i] /
                                std::max(1e-6, m_cfg.v2ActivityProb);
            }
        }

        // Handover: a serving-cell change between two campus cells (attach/detach
        // to/from off-campus, cell 0, is not a handover).
        if (!m_first && cur != m_prevCell[i] && cur != 0 && m_prevCell[i] != 0)
        {
            hoOut[m_prevCell[i]]++;
            hoIn[cur]++;
        }
        m_prevCell[i] = cur;
    }
    m_first = false;

    for (uint32_t c = 1; c <= m_nCells; ++c)
    {
        double meanSinrDb =
            (ueCount[c] > 0) ? 10.0 * std::log10(sinrLin[c] / ueCount[c]) : 0.0;
        m_out << minute << "," << c << "," << m_cellPos[c].x << "," << m_cellPos[c].y << ","
              << ueCount[c] << "," << isacCount[c] << "," << std::setprecision(3) << meanSinrDb
              << "," << hoIn[c] << "," << hoOut[c] << "," << std::setprecision(4) << traffic[c]
              << "," << presence[c] << "\n";
    }

    if (minute < m_cfg.datasetMinutes)
    {
        Simulator::Schedule(m_interval, &DatasetCollector::Sample, this);
    }
}

/** Antenna levels the RL policy can choose from. */
static const uint32_t RL_ANTENNA_CHOICES[] = {2, 4, 8, 16};
/** Number of entries in RL_ANTENNA_CHOICES. */
static const uint32_t RL_ANTENNA_NCHOICES = 4;
/** Observation values published per cell. */
static const uint32_t RL_OBS_PER_CELL = 10;

/**
 * RL antenna-control environment: exposes the campus scenario to a Python
 * agent through the ns3-ai Gym interface.
 *
 * One step is one simulated minute. Every step publishes the per-cell
 * observation, receives one antenna level per cell, and applies it immediately
 * through CoverageModel::SetAntennas, so the next minute is simulated under the
 * new coverage. Stepping every minute (rather than every control period) keeps
 * the observation stream at the same resolution the TCN was trained on; holding
 * an action constant over a control period is done on the Python side.
 *
 * Observation (float, length N*10+1), per cell c = 1..N:
 *   [traffic_demand_mbps, ue_count, isac_count, presence_count,
 *    mean_sinr_db, ho_in, ho_out, antenna_count, outage_ue, outage_traffic_mbps]
 * followed by the current minute as the final element. A connected UE whose SINR is
 * below SimConfig::rlSinrThresholdDb, or outside every cell's current coverage radius, is
 * still counted in ue_count and in the demand, and additionally in outage_ue /
 * outage_traffic_mbps (its demand is unserved). UEs beyond the largest radius of every
 * cell are outside the service area and never counted as outages.
 *
 * Extra info (string): "minute=M;spm=S;conn=C;out=O;ev=E;evl=L", campus totals over the
 * outage samples since the previous step. The minute is sampled S times (rlOutageSamplesPerMin,
 * every 60/S seconds; the last sample is the step itself): C = connected UE samples, O = those
 * in outage, E = outage runs started, L = runs that reached rlOutageEventSec, where a run lasts
 * (its samples) x (60/S) s; L = -1 when 60/S >= rlOutageEventSec (not measurable). The sub-minute
 * samples use the same outage rule as the step and draw no random numbers, so the
 * observation is the same for every S up to rounding: the waypoint model advances a position
 * by velocity x elapsed time at each query, so extra queries change the last bits of later
 * positions (about 1e-12 m). Only campus totals are sent because the ns3-ai
 * message buffer (MSG_BUFFER_SIZE) has no room for per-cell values.
 *
 * Action (uint32_t, length N): index into RL_ANTENNA_CHOICES for each cell,
 * where element i addresses cell id i+1.
 *
 * GetReward always returns 0: the reward is computed on the Python side so that
 * its shaping can be changed without recompiling ns-3.
 */
class RlAntennaEnv : public OpenGymEnv
{
  public:
    /**
     * Constructor.
     *
     * @param cfg Simulation configuration (must outlive this object).
     */
    explicit RlAntennaEnv(const SimConfig& cfg)
        : m_cfg(cfg),
          m_interval(Seconds(1.0))
    {
    }

    /**
     * Register this type.
     *
     * @return the object TypeId.
     */
    static TypeId GetTypeId()
    {
        static TypeId tid =
            TypeId("ns3::RlAntennaEnv").SetParent<OpenGymEnv>().SetGroupName("OpenGym");
        return tid;
    }

    /** Build the scenario, then step it minute-by-minute under agent control. */
    void Run();

    Ptr<OpenGymSpace> GetActionSpace() override;
    Ptr<OpenGymSpace> GetObservationSpace() override;
    bool GetGameOver() override;
    Ptr<OpenGymDataContainer> GetObservation() override;
    float GetReward() override;
    std::string GetExtraInfo() override;
    bool ExecuteActions(Ptr<OpenGymDataContainer> action) override;

  private:
    /** Observe one minute, hand it to the agent and apply the reply. */
    void Step();

    /**
     * Sample the outage state of every UE between two steps. Draws no random numbers,
     * so the step observations depend on how often this runs only through the rounding of
     * the positions it queries.
     */
    void SubSample();

    /**
     * Classify a connected UE at the current sim time.
     *
     * @param mob The UE mobility model.
     * @param sinrDb Set to the wideband SINR (dB) toward the serving cell: the strongest
     *        covering cell, or the strongest cell overall when none covers the UE. Such an
     *        uncovered UE inside the service area is attributed to its nearest cell instead,
     *        so the returned cell can differ from the one this SINR refers to.
     * @param dropped Set to whether the UE is in outage: inside the service area but outside
     *        every cell's coverage radius, or below the SINR threshold.
     * @return the cell id the UE is attributed to.
     */
    uint32_t ClassifyConnected(Ptr<MobilityModel> mob, double& sinrDb, bool& dropped) const;

    /**
     * Add one outage sample of UE @p i to the campus outage counters.
     *
     * @param i UE index.
     * @param connected Whether the UE is connected at this sample.
     * @param dropped Whether the connected UE is in outage at this sample.
     */
    void AddOutageSample(uint32_t i, bool connected, bool dropped);

    /** Whether UE @p i is on campus at the current sim time (step function). */
    bool OnNow(uint32_t i) const;

    const SimConfig& m_cfg;                     ///< Simulation configuration
    Time m_interval;                            ///< Step period (1 min)
    NodeContainer m_gnbs;                       ///< gNB nodes
    NodeContainer m_ues;                        ///< Waypoint UE nodes
    std::vector<uint32_t> m_cellOfGnb;          ///< gNB index -> cell id (1..N)
    std::vector<Vector> m_cellPos;              ///< cell id -> gNB position (1..N)
    std::vector<uint32_t> m_antennas;           ///< cell id -> current antenna count
    std::vector<std::vector<std::pair<double, bool>>> m_ueOn; ///< per-UE on-campus schedule
    std::vector<uint32_t> m_prevCell;           ///< per-UE serving cell last minute (0=none)
    std::unique_ptr<CoverageModel> m_coverage;  ///< Antenna-aware coverage / SINR
    uint32_t m_nCells{0};                       ///< Number of cells
    bool m_first{true};                         ///< First step (skip handover counting)
    uint32_t m_minute{0};                       ///< Current minute index

    // Per-UE stochastic state as in DatasetCollector. Times are sim seconds (1 s = 1 min).
    std::vector<double> m_rateMult;             ///< per-UE demand multiplier
    std::vector<double> m_connDelay;            ///< per-UE connection delay (min)
    std::vector<double> m_ueArm;                ///< per-UE last arrival + building lead (min)
    std::vector<bool> m_prevOn;                 ///< per-UE previous-minute on-campus flag
    std::vector<double> m_baseRate;             ///< cell -> per-person demand (Mbps)
    std::vector<double> m_eveMult;              ///< cell -> evening (>=18:00) multiplier
    std::vector<double> m_lagMin;               ///< cell -> arrival lead over traffic (min)
    Ptr<UniformRandomVariable> m_rng;           ///< RNG for activity / detection

    // Latest observation, filled by Step() and read by GetObservation().
    std::vector<uint32_t> m_ueCount;            ///< cell -> connected UEs
    std::vector<uint32_t> m_isacCount;          ///< cell -> ISAC detections
    std::vector<uint32_t> m_presence;           ///< cell -> people within sensing range
    std::vector<uint32_t> m_hoIn;               ///< cell -> handovers in
    std::vector<uint32_t> m_hoOut;              ///< cell -> handovers out
    std::vector<double> m_traffic;              ///< cell -> demanded traffic (Mbps)
    std::vector<double> m_meanSinrDb;           ///< cell -> mean SINR (dB)
    std::vector<uint32_t> m_outageUe;           ///< cell -> connected UEs below SINR threshold
    std::vector<double> m_outageTraffic;        ///< cell -> demand of those UEs (Mbps)

    // Campus outage samples since the last published step (see the class description).
    uint64_t m_sampConnected{0};                ///< connected UE samples
    uint64_t m_sampOutage{0};                   ///< connected UE samples in outage
    uint64_t m_sampEvents{0};                   ///< outage runs started
    uint64_t m_sampLongEvents{0};               ///< outage runs reaching rlOutageEventSec
    std::vector<double> m_outRunSec;            ///< per-UE current outage run length (s)
};

bool
RlAntennaEnv::OnNow(uint32_t i) const
{
    if (i >= m_ueOn.size() || m_ueOn[i].empty())
    {
        return true;
    }
    const double now = Simulator::Now().GetSeconds();
    bool on = m_ueOn[i].front().second;
    for (const auto& e : m_ueOn[i])
    {
        if (e.first <= now)
        {
            on = e.second;
        }
        else
        {
            break;
        }
    }
    return on;
}

Ptr<OpenGymSpace>
RlAntennaEnv::GetActionSpace()
{
    std::vector<uint32_t> shape = {m_nCells};
    return CreateObject<OpenGymBoxSpace>(0,
                                         RL_ANTENNA_NCHOICES - 1,
                                         shape,
                                         TypeNameGet<uint32_t>());
}

Ptr<OpenGymSpace>
RlAntennaEnv::GetObservationSpace()
{
    std::vector<uint32_t> shape = {m_nCells * RL_OBS_PER_CELL + 1};
    return CreateObject<OpenGymBoxSpace>(-1e6, 1e6, shape, TypeNameGet<float>());
}

bool
RlAntennaEnv::GetGameOver()
{
    return false;
}

Ptr<OpenGymDataContainer>
RlAntennaEnv::GetObservation()
{
    std::vector<uint32_t> shape = {m_nCells * RL_OBS_PER_CELL + 1};
    Ptr<OpenGymBoxContainer<float>> box = CreateObject<OpenGymBoxContainer<float>>(shape);
    for (uint32_t c = 1; c <= m_nCells; ++c)
    {
        box->AddValue(static_cast<float>(m_traffic[c]));
        box->AddValue(static_cast<float>(m_ueCount[c]));
        box->AddValue(static_cast<float>(m_isacCount[c]));
        box->AddValue(static_cast<float>(m_presence[c]));
        box->AddValue(static_cast<float>(m_meanSinrDb[c]));
        box->AddValue(static_cast<float>(m_hoIn[c]));
        box->AddValue(static_cast<float>(m_hoOut[c]));
        box->AddValue(static_cast<float>(m_antennas[c]));
        box->AddValue(static_cast<float>(m_outageUe[c]));
        box->AddValue(static_cast<float>(m_outageTraffic[c]));
    }
    box->AddValue(static_cast<float>(m_minute));
    return box;
}

float
RlAntennaEnv::GetReward()
{
    return 0.0;
}

std::string
RlAntennaEnv::GetExtraInfo()
{
    // A run lasts (its samples) x (sample period), so with a period at least rlOutageEventSec
    // every run would count as long; -1 marks the long-event count as not measurable.
    const bool longMeasurable = 60.0 / m_cfg.rlOutageSamplesPerMin < m_cfg.rlOutageEventSec;
    std::ostringstream os;
    os << "minute=" << m_minute << ";spm=" << m_cfg.rlOutageSamplesPerMin
       << ";conn=" << m_sampConnected << ";out=" << m_sampOutage << ";ev=" << m_sampEvents
       << ";evl=" << (longMeasurable ? static_cast<int64_t>(m_sampLongEvents) : -1);
    return os.str();
}

bool
RlAntennaEnv::ExecuteActions(Ptr<OpenGymDataContainer> action)
{
    Ptr<OpenGymBoxContainer<uint32_t>> box = DynamicCast<OpenGymBoxContainer<uint32_t>>(action);
    if (!box)
    {
        return false;
    }
    for (uint32_t c = 1; c <= m_nCells; ++c)
    {
        uint32_t idx = box->GetValue(c - 1);
        // A policy bug must not corrupt the run, so an out-of-range index is clamped.
        if (idx >= RL_ANTENNA_NCHOICES)
        {
            idx = RL_ANTENNA_NCHOICES - 1;
        }
        m_antennas[c] = RL_ANTENNA_CHOICES[idx];
    }
    m_coverage->SetAntennas(m_antennas);
    return true;
}

void
RlAntennaEnv::Run()
{
    std::vector<bool> siteIndoor;
    std::vector<Vector> gnbPos = LoadSitePositions(m_cfg.sitesFile, 10.0, &siteIndoor);
    const uint32_t nGnb = gnbPos.size();
    NS_ABORT_MSG_IF(nGnb == 0, "rlEnv mode needs a --sitesFile");
    NS_ABORT_MSG_IF(m_cfg.waypoints.empty(), "rlEnv mode needs --waypoints");
    NS_ABORT_MSG_IF(m_cfg.rlOutageSamplesPerMin == 0, "rlOutageSamplesPerMin must be >= 1");
    m_nCells = nGnb;

    // Stable cell ids by real (y, then x): the same numbering the dataset uses.
    std::vector<uint32_t> order(nGnb);
    for (uint32_t i = 0; i < nGnb; ++i)
    {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return (gnbPos[a].y != gnbPos[b].y) ? (gnbPos[a].y < gnbPos[b].y)
                                            : (gnbPos[a].x < gnbPos[b].x);
    });
    m_cellOfGnb.assign(nGnb, 0);
    m_cellPos.assign(nGnb + 1, Vector());
    for (uint32_t rank = 0; rank < nGnb; ++rank)
    {
        m_cellOfGnb[order[rank]] = rank + 1;
        m_cellPos[rank + 1] = gnbPos[order[rank]];
    }

    m_gnbs.Create(nGnb);
    MobilityHelper gnbMob;
    Ptr<ListPositionAllocator> gnbAlloc = CreateObject<ListPositionAllocator>();
    for (const auto& p : gnbPos)
    {
        gnbAlloc->Add(p);
    }
    gnbMob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    gnbMob.SetPositionAllocator(gnbAlloc);
    gnbMob.Install(m_gnbs);

    m_antennas.assign(nGnb + 1, m_cfg.rlAntennaInit);
    m_coverage =
        std::make_unique<CoverageModel>(m_cfg.slmCarrierFrequency, m_gnbs, m_cellOfGnb, m_antennas);
    m_coverage->SetBuildingEntryLoss(IndoorSitePositions(gnbPos, siteIndoor),
                                     m_cfg.o2iIndoorRadius,
                                     m_cfg.o2iLossDb);

    uint32_t nUe = LoadWaypointUes(m_cfg.waypoints, m_ues, m_ueOn);
    NS_ABORT_MSG_IF(nUe == 0, "rlEnv mode: no UEs in waypoints file");
    m_prevCell.assign(nUe, 0);
    m_outRunSec.assign(nUe, 0.0);

    auto rates = LoadRates(m_cfg.ratesFile, nGnb);
    m_baseRate.assign(nGnb + 1, 0.5);
    m_eveMult.assign(nGnb + 1, 1.0);
    m_lagMin.assign(nGnb + 1, 0.0);
    for (uint32_t c = 1; c <= nGnb; ++c)
    {
        m_baseRate[c] = rates[c].baseRate;
        m_eveMult[c] = rates[c].eveningMult;
        m_lagMin[c] = rates[c].lagMin;
    }
    m_rng = CreateObject<UniformRandomVariable>();
    Ptr<UniformRandomVariable> init = CreateObject<UniformRandomVariable>();
    m_rateMult.assign(nUe, 1.0);
    m_connDelay.assign(nUe, 0.0);
    m_ueArm.assign(nUe, -1e12);
    m_prevOn.assign(nUe, false);
    const double lo = std::max(0.1, 1.0 - m_cfg.v2RateCv);
    const double hi = 1.0 + m_cfg.v2RateCv;
    for (uint32_t i = 0; i < nUe; ++i)
    {
        m_rateMult[i] = init->GetValue(lo, hi);
        // The sim clock runs one second per minute, so minutes are used as-is.
        m_connDelay[i] = init->GetValue(0.0, 2.0 * m_cfg.v2ConnDelayMean);
    }

    m_meanSinrDb.assign(nGnb + 1, 0.0);
    m_ueCount.assign(nGnb + 1, 0);
    m_isacCount.assign(nGnb + 1, 0);
    m_presence.assign(nGnb + 1, 0);
    m_hoIn.assign(nGnb + 1, 0);
    m_hoOut.assign(nGnb + 1, 0);
    m_traffic.assign(nGnb + 1, 0.0);
    m_outageUe.assign(nGnb + 1, 0);
    m_outageTraffic.assign(nGnb + 1, 0.0);

    // Must match shm_names() in rl-env_v4/campus_env.py. Set before the first
    // Notify, which is when the gym interface opens the shared memory.
    if (!m_cfg.rlShmTag.empty())
    {
        const std::string& tag = m_cfg.rlShmTag;
        Ns3AiMsgInterface::Get()->SetNames("ns3ai seg " + tag,
                                           "ns3ai cpp2py " + tag,
                                           "ns3ai py2cpp " + tag,
                                           "ns3ai lockable " + tag);
    }

    // Registered only once the scenario is complete, so the first space query
    // from Python already sees the real cell count.
    SetOpenGymInterface(OpenGymInterface::Get());

    Simulator::Schedule(Seconds(0.0), &RlAntennaEnv::Step, this);
    Simulator::Stop(Seconds(static_cast<double>(m_cfg.datasetMinutes) + 0.5));
    Simulator::Run();
    NotifySimulationEnd();
    Simulator::Destroy();
}

void
RlAntennaEnv::Step()
{
    const double now = Simulator::Now().GetSeconds();
    m_minute = static_cast<uint32_t>(std::llround(now));
    const double hour = 9.0 + now / 60.0;

    std::fill(m_ueCount.begin(), m_ueCount.end(), 0);
    std::fill(m_isacCount.begin(), m_isacCount.end(), 0);
    std::fill(m_presence.begin(), m_presence.end(), 0);
    std::fill(m_hoIn.begin(), m_hoIn.end(), 0);
    std::fill(m_hoOut.begin(), m_hoOut.end(), 0);
    std::fill(m_traffic.begin(), m_traffic.end(), 0.0);
    std::fill(m_outageUe.begin(), m_outageUe.end(), 0);
    std::fill(m_outageTraffic.begin(), m_outageTraffic.end(), 0.0);
    std::vector<double> sinrLin(m_nCells + 1, 0.0);

    for (uint32_t i = 0; i < m_ues.GetN(); ++i)
    {
        Ptr<MobilityModel> mob = m_ues.Get(i)->GetObject<MobilityModel>();
        const bool on = OnNow(i);
        uint32_t cur = 0;

        // Arrival resets the connection-delay clock, which starts after the lead of the
        // building the UE arrives at (same rule as DatasetCollector).
        if (on && !m_prevOn[i])
        {
            double d = 0.0;
            m_ueArm[i] =
                now + m_lagMin[m_coverage->NearestCellWithin(mob,
                                                             std::numeric_limits<double>::max(),
                                                             d)];
        }
        m_prevOn[i] = on;

        if (on)
        {
            double ds = 0.0;
            uint32_t sc = m_coverage->NearestCellWithin(mob, m_cfg.isacSenseRadius, ds);
            if (sc != 0)
            {
                m_presence[sc]++;
                if (m_rng->GetValue() < m_cfg.v2Pd)
                {
                    m_isacCount[sc]++;
                }
            }
        }

        const bool connected = on && (now >= m_ueArm[i] + m_connDelay[i]);
        bool dropped = false;
        if (connected)
        {
            double sinrDb = 0.0;
            cur = ClassifyConnected(mob, sinrDb, dropped);
            m_ueCount[cur]++;
            sinrLin[cur] += std::pow(10.0, sinrDb / 10.0);
            if (dropped)
            {
                m_outageUe[cur]++;
            }
            if (m_rng->GetValue() < m_cfg.v2ActivityProb)
            {
                double eve = (hour >= 18.0) ? m_eveMult[cur] : 1.0;
                double demand = m_baseRate[cur] * eve * m_rateMult[i] /
                                std::max(1e-6, m_cfg.v2ActivityProb);
                m_traffic[cur] += demand;
                if (dropped)
                {
                    m_outageTraffic[cur] += demand;
                }
            }
        }
        AddOutageSample(i, connected, dropped);

        if (!m_first && cur != m_prevCell[i] && cur != 0 && m_prevCell[i] != 0)
        {
            m_hoOut[m_prevCell[i]]++;
            m_hoIn[cur]++;
        }
        m_prevCell[i] = cur;
    }
    m_first = false;

    for (uint32_t c = 1; c <= m_nCells; ++c)
    {
        m_meanSinrDb[c] =
            (m_ueCount[c] > 0) ? 10.0 * std::log10(sinrLin[c] / m_ueCount[c]) : 0.0;
    }

    // Blocks until the agent replies; ExecuteActions applies the new antennas.
    Notify();
    m_sampConnected = 0;
    m_sampOutage = 0;
    m_sampEvents = 0;
    m_sampLongEvents = 0;

    if (m_minute < m_cfg.datasetMinutes)
    {
        const uint32_t spm = m_cfg.rlOutageSamplesPerMin;
        for (uint32_t k = 1; k < spm; ++k)
        {
            Simulator::Schedule(Seconds(m_interval.GetSeconds() * k / spm),
                                &RlAntennaEnv::SubSample,
                                this);
        }
        Simulator::Schedule(m_interval, &RlAntennaEnv::Step, this);
    }
}

void
RlAntennaEnv::SubSample()
{
    const double now = Simulator::Now().GetSeconds();
    for (uint32_t i = 0; i < m_ues.GetN(); ++i)
    {
        // Arrivals are armed by Step(), so a UE that came on campus after the last step is
        // not connected yet, as it would not be at the next step either.
        const bool on = OnNow(i) && m_prevOn[i];
        const bool connected = on && (now >= m_ueArm[i] + m_connDelay[i]);
        bool dropped = false;
        if (connected)
        {
            double sinrDb = 0.0;
            ClassifyConnected(m_ues.Get(i)->GetObject<MobilityModel>(), sinrDb, dropped);
        }
        AddOutageSample(i, connected, dropped);
    }
}

uint32_t
RlAntennaEnv::ClassifyConnected(Ptr<MobilityModel> mob, double& sinrDb, bool& dropped) const
{
    bool covered = false;
    bool inServiceArea = false;
    uint32_t cur = m_coverage->ServingCellSinrInCoverage(mob, sinrDb, covered, inServiceArea);
    // Shrinking a cell also lowers its gain, so the strongest cell for a UE it no
    // longer covers is often a neighbor. Charging the outage there would reward
    // the shrinking cell; the nearest cell is the one that would serve the UE at
    // full antennas, so it carries the outage instead.
    if (!covered && inServiceArea)
    {
        double d = 0.0;
        cur = m_coverage->NearestCellWithin(mob, std::numeric_limits<double>::max(), d);
    }
    // A UE no cell could reach even at full antennas (e.g. walking off campus) is
    // outside the service area: no control decision could serve it, so it is not
    // charged as an outage. It still counts toward demand and ue_count.
    dropped = inServiceArea && (!covered || sinrDb < m_cfg.rlSinrThresholdDb);
    return cur;
}

void
RlAntennaEnv::AddOutageSample(uint32_t i, bool connected, bool dropped)
{
    if (!connected || !dropped)
    {
        m_sampConnected += connected ? 1 : 0;
        m_outRunSec[i] = 0.0;
        return;
    }
    m_sampConnected++;
    m_sampOutage++;
    const double before = m_outRunSec[i];
    const double after = before + 60.0 / m_cfg.rlOutageSamplesPerMin;
    if (before == 0.0)
    {
        m_sampEvents++;
    }
    // The tolerance keeps a sample period that does not divide the threshold evenly
    // (e.g. 60/7 s) from missing it by a rounding error.
    const double longSec = m_cfg.rlOutageEventSec - 1e-9;
    if ((before == 0.0 || before < longSec) && after >= longSec)
    {
        m_sampLongEvents++;
    }
    m_outRunSec[i] = after;
}

} // namespace

int
main(int argc, char* argv[])
{
    SimConfig cfg;
    cfg.Parse(argc, argv);

    // RL mode: expose the scenario as an ns3-ai Gym environment so a Python
    // agent drives the per-cell antenna level.
    if (cfg.rlEnv)
    {
        Ptr<RlAntennaEnv> rlEnv = CreateObject<RlAntennaEnv>(cfg);
        rlEnv->Run();
        return EXIT_SUCCESS;
    }

    // Dataset mode: step the scenario minute-by-minute (waypoint UEs) and record
    // per-cell RF observations for the TCN training data.
    NS_ABORT_MSG_IF(!cfg.dataset, "Choose a mode: --dataset or --rlEnv");
    DatasetCollector datasetCollector(cfg);
    datasetCollector.Run();
    return EXIT_SUCCESS;
}
