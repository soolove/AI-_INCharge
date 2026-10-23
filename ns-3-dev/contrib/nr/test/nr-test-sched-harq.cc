// Copyright (c) 2025 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
//
// SPDX-License-Identifier: GPL-2.0-only

#include "ns3/config.h"
#include "ns3/nr-mac-sched-sap.h"
#include "ns3/nr-mac-scheduler-ns3.h"
#include "ns3/nr-mac-scheduler-ofdma-rr.h"
#include "ns3/nr-mac-scheduler-ofdma-symbol-per-beam.h"
#include "ns3/nr-mac-scheduler-tdma-rr.h"
#include "ns3/nr-spectrum-phy.h"
#include "ns3/object-factory.h"
#include "ns3/test.h"

/**
 * @file nr-test-sched-harq.cc
 * @ingroup test
 *
 * @brief This file contain tests for the round-robin nature of nr-mac-scheduler-harq-rr.
 * It also tests if allocations are properly consolidated to use less symbols,
 * and maintain or increase MCS, in order to increase the change of successful decoding.
 *
 */
namespace ns3
{
class TestCschedSapUserHarq : public NrMacCschedSapUser
{
  public:
    TestCschedSapUserHarq()
        : NrMacCschedSapUser()
    {
    }

    void CschedCellConfigCnf(
        [[maybe_unused]] const struct CschedCellConfigCnfParameters& params) override
    {
    }

    void CschedUeConfigCnf(
        [[maybe_unused]] const struct CschedUeConfigCnfParameters& params) override
    {
    }

    void CschedLcConfigCnf(
        [[maybe_unused]] const struct CschedLcConfigCnfParameters& params) override
    {
    }

    void CschedLcReleaseCnf(
        [[maybe_unused]] const struct CschedLcReleaseCnfParameters& params) override
    {
    }

    void CschedUeReleaseCnf(
        [[maybe_unused]] const struct CschedUeReleaseCnfParameters& params) override
    {
    }

    void CschedUeConfigUpdateInd(
        [[maybe_unused]] const struct CschedUeConfigUpdateIndParameters& params) override
    {
    }

    void CschedCellConfigUpdateInd(
        [[maybe_unused]] const struct CschedCellConfigUpdateIndParameters& params) override
    {
    }
};

class TestSchedSapUserHarq : public NrMacSchedSapUser
{
  public:
    TestSchedSapUserHarq(
        std::function<void(const struct SchedConfigIndParameters&)> schedConfigIndCallback =
            [](auto params) {},
        std::function<uint32_t(void)> symbolsPerSlotCallback = []() { return 14; })
        : NrMacSchedSapUser(),
          m_schedConfigIndCallback(schedConfigIndCallback),
          m_symbolsPerSlotCallback(symbolsPerSlotCallback)
    {
    }

    void SchedConfigInd(const NrMacSchedSapUser::SchedConfigIndParameters& params) override
    {
        if (m_schedConfigIndCallback)
        {
            m_schedConfigIndCallback(params);
        }
    }

    // For the rest, setup some hard-coded values; for the moment, there is
    // no need to have real values here.
    Ptr<const SpectrumModel> GetSpectrumModel() const override
    {
        return nullptr;
    }

    uint32_t GetNumRbPerRbg() const override
    {
        return 1;
    }

    uint8_t GetNumHarqProcess() const override
    {
        return 20;
    }

    uint16_t GetBwpId() const override
    {
        return 0;
    }

    uint16_t GetCellId() const override
    {
        return 0;
    }

    uint32_t GetSymbolsPerSlot() const override
    {
        return m_symbolsPerSlotCallback();
    }

    Time GetSlotPeriod() const override
    {
        return MilliSeconds(1);
    }

    void BuildRarList(SlotAllocInfo& allocInfo) override
    {
    }

  private:
    std::function<void(const NrMacSchedSapUser::SchedConfigIndParameters&)>
        m_schedConfigIndCallback;
    std::function<uint32_t()> m_symbolsPerSlotCallback;
};

/**
 * @brief TestSched testcase
 */
class NrTestMacSchedulerHarqRrReshape : public TestCase
{
  public:
    /**
     * @brief Create NrSchedGeneralTestCase
     * @param scheduler Scheduler to test
     * @param name Name of the test
     */
    NrTestMacSchedulerHarqRrReshape(std::vector<DciInfoElementTdma> dcis,
                                    uint8_t startingSymbol,
                                    uint8_t numSymbols,
                                    const std::string& name)
        : TestCase(name),
          m_dcis(dcis),
          m_startingSymbol(startingSymbol),
          m_numSymbols(numSymbols)
    {
    }

  protected:
    void DoRun() override;
    const std::vector<DciInfoElementTdma> m_dcis;
    const uint8_t m_startingSymbol;
    const uint8_t m_numSymbols;
};

void
NrTestMacSchedulerHarqRrReshape::DoRun()
{
    // Prepare common settings for both TDMA and OFDMA schedulers
    auto cellConfig = NrMacCschedSapProvider::CschedCellConfigReqParameters();
    cellConfig.m_dlBandwidth = 10; // 10 RBGs
    cellConfig.m_ulBandwidth = 10;

    std::vector<NrMacCschedSapProvider::CschedUeConfigReqParameters> ueConfig;
    for (const auto& dci : m_dcis)
    {
        NrMacCschedSapProvider::CschedUeConfigReqParameters config{};
        config.m_rnti = dci.m_rnti;
        config.m_transmissionMode = 0;
        config.m_beamId = BeamId(dci.m_rnti / 5, 0);
        ueConfig.push_back(config);
    }

    auto* schedSapUser = new TestSchedSapUserHarq();
    auto* cschedSapUser = new TestCschedSapUserHarq();

    for (bool isTdma : {true, false})
    {
        Ptr<NrMacSchedulerNs3> scheduler;
        if (isTdma)
        {
            scheduler = CreateObject<NrMacSchedulerTdmaRR>();
        }
        else
        {
            scheduler = CreateObject<NrMacSchedulerOfdmaRR>();
        }
        scheduler->SetMacSchedSapUser(schedSapUser);
        scheduler->SetMacCschedSapUser(cschedSapUser);
        scheduler->DoCschedCellConfigReq(cellConfig);
        for (const auto& ueConf : ueConfig)
        {
            scheduler->DoCschedUeConfigReq(ueConf);
        }
        const bool isDl = true;
        std::vector<bool> bitmask(10, true);
        auto startingSymbol = m_startingSymbol;
        auto numSymbols = m_numSymbols;
        auto reshapedDcisTdma =
            scheduler->ReshapeAllocation(m_dcis, startingSymbol, numSymbols, bitmask, isDl);

        // Check if we went above number of available symbols
        auto reshapedAllocatedSymbols = 0;
        if (!reshapedDcisTdma.empty())
        {
            auto smallestStartSymbol = std::numeric_limits<uint8_t>::max();
            auto largestFinalSymbol = m_startingSymbol;
            for (auto& dci : reshapedDcisTdma)
            {
                if (smallestStartSymbol > dci.m_symStart)
                {
                    smallestStartSymbol = dci.m_symStart;
                }
                if (largestFinalSymbol < (dci.m_symStart + dci.m_numSym))
                {
                    largestFinalSymbol = dci.m_symStart + dci.m_numSym;
                }
            }
            NS_ASSERT_MSG(smallestStartSymbol != std::numeric_limits<uint8_t>::max(),
                          "There must have been a valid starting symbol");
            reshapedAllocatedSymbols = largestFinalSymbol - smallestStartSymbol;
        }
        NS_TEST_ASSERT_MSG_LT_OR_EQ(
            reshapedAllocatedSymbols,
            +m_numSymbols,
            (isTdma ? "TDMA" : "OFDMA")
                << ": Reshaped unexpectedly into more symbols than available");

        // If the test case has no symbols, do not continue which checks, because the one above
        // should suffice
        if (m_numSymbols == 0)
        {
            continue;
        }

        // If there is no reshaped DCI, we do not continue checks
        // (temporary until reshape can handle multiple DCIs, and later multiple beams)
        if (reshapedDcisTdma.empty())
        {
            continue;
        }

        // Test we haven't changed what we are not supposed to change
        for (auto& reshapedDci : reshapedDcisTdma)
        {
            const auto originalDci =
                std::find_if(m_dcis.begin(), m_dcis.end(), [=](DciInfoElementTdma a) {
                    return reshapedDci.m_rnti == a.m_rnti;
                });
            NS_TEST_EXPECT_MSG_EQ((originalDci != m_dcis.end()),
                                  true,
                                  "Reshaped allocation changed DCI RNTI");
            NS_TEST_ASSERT_MSG_EQ(reshapedDci.m_format,
                                  originalDci->m_format,
                                  "Reshaped allocation changed DCI format");
            NS_TEST_ASSERT_MSG_EQ(+reshapedDci.m_mcs,
                                  +originalDci->m_mcs,
                                  "Reshaped allocation changed DCI MCS");
            NS_TEST_ASSERT_MSG_EQ(+reshapedDci.m_rank,
                                  +originalDci->m_rank,
                                  "Reshaped allocation changed DCI rank");
            NS_TEST_ASSERT_MSG_EQ(reshapedDci.m_precMats,
                                  originalDci->m_precMats,
                                  "Reshaped allocation changed DCI Precoding matrices");
            NS_TEST_ASSERT_MSG_EQ(reshapedDci.m_tbSize,
                                  originalDci->m_tbSize,
                                  "Reshaped allocation changed DCI TBS");
            NS_TEST_ASSERT_MSG_EQ(reshapedDci.m_ndi,
                                  originalDci->m_ndi,
                                  "Reshaped allocation changed DCI NDI");
            NS_TEST_ASSERT_MSG_EQ(reshapedDci.m_rv,
                                  originalDci->m_rv,
                                  "Reshaped allocation changed DCI HARQ RV");
            NS_TEST_ASSERT_MSG_EQ(reshapedDci.m_type,
                                  originalDci->m_type,
                                  "Reshaped allocation changed DCI type");
            NS_TEST_ASSERT_MSG_EQ(reshapedDci.m_bwpIndex,
                                  originalDci->m_bwpIndex,
                                  "Reshaped allocation changed DCI BWP index");
            NS_TEST_ASSERT_MSG_EQ(reshapedDci.m_tpc,
                                  originalDci->m_tpc,
                                  "Reshaped allocation changed DCI TPC");

            // Test if we changed what we are supposed to change
            if (isTdma)
            {
                NS_TEST_ASSERT_MSG_GT_OR_EQ(
                    +std::count(reshapedDci.m_rbgBitmask.begin(),
                                reshapedDci.m_rbgBitmask.end(),
                                true),
                    +std::count(originalDci->m_rbgBitmask.begin(),
                                originalDci->m_rbgBitmask.end(),
                                true),
                    "Reshaped TDMA allocation unexpectedly has less RBGs than the original");
                NS_TEST_ASSERT_MSG_LT_OR_EQ(
                    +reshapedDci.m_numSym,
                    +originalDci->m_numSym,
                    "Reshaped TDMA allocation unexpectedly has more symbols than the original");
                NS_TEST_ASSERT_MSG_GT_OR_EQ(
                    reshapedDci.m_numSym * std::count(reshapedDci.m_rbgBitmask.begin(),
                                                      reshapedDci.m_rbgBitmask.end(),
                                                      true),
                    originalDci->m_numSym * std::count(originalDci->m_rbgBitmask.begin(),
                                                       originalDci->m_rbgBitmask.end(),
                                                       true),
                    "Reshaped TDMA allocation unexpectedly has less resources than the original");
            }
            else
            {
            }
        }
    }
    delete schedSapUser;
    delete cschedSapUser;
}

class NrTestMacSchedulerHarqRrScheduleDlHarq : public NrTestMacSchedulerHarqRrReshape
{
  public:
    /**
     * @brief Create NrSchedGeneralTestCase
     * @param scheduler Scheduler to test
     * @param name Name of the test
     */
    NrTestMacSchedulerHarqRrScheduleDlHarq(std::vector<DciInfoElementTdma> dcis,
                                           uint8_t startingSymbol,
                                           uint8_t numSymbols,
                                           const std::string& name)
        : NrTestMacSchedulerHarqRrReshape(dcis, startingSymbol, numSymbols, name)
    {
    }

    void CheckSchedule(const NrMacSchedSapUser::SchedConfigIndParameters& params);

  protected:
    void DoRun() override;
    bool m_testingTdma{false};
};

void
NrTestMacSchedulerHarqRrScheduleDlHarq::CheckSchedule(
    const NrMacSchedSapUser::SchedConfigIndParameters& params)
{
    // Retrieve resulting scheduled HARQ DCIs
    std::vector<DciInfoElementTdma> resultingDcis;
    for (const auto& varTtiAllocInfo : params.m_slotAllocInfo.m_varTtiAllocInfo)
    {
        // We only want data, no control DCIs
        if (varTtiAllocInfo.m_dci->m_type == DciInfoElementTdma::DATA)
        {
            resultingDcis.push_back(*varTtiAllocInfo.m_dci);
        }
    }

    if (m_testingTdma)
    {
        // TDMA should minimize symbols used by HARQ allocations, to fit more HARQ retransmissions
        // in a slot
        for (auto& resultingDci : resultingDcis)
        {
            auto& originalDci =
                *std::find_if(m_dcis.begin(), m_dcis.end(), [resultingDci](auto& a) {
                    return a.m_rnti == resultingDci.m_rnti;
                });
            NS_TEST_EXPECT_MSG_LT_OR_EQ(+resultingDci.m_numSym,
                                        +originalDci.m_numSym,
                                        "Number of symbols for TDMA should be same or smaller");

            auto originalRbgs =
                std::count(originalDci.m_rbgBitmask.begin(), originalDci.m_rbgBitmask.end(), true);
            auto resultingRbgs = std::count(resultingDci.m_rbgBitmask.begin(),
                                            resultingDci.m_rbgBitmask.end(),
                                            true);
            NS_TEST_ASSERT_MSG_EQ(resultingRbgs * resultingDci.m_numSym,
                                  originalRbgs * originalDci.m_numSym,
                                  "Number of allocated resources should not change (error model "
                                  "assumes it remains constant)");
        }
    }
    // Testing OFDMA
    else
    {
        // OFDMA should maximize symbols used by HARQ retransmissions, in order to make better use
        // of RBGs for other retransmissions. However, the total number of resources of
        // retransmissions of a symbol should use the least amount of symbols possible, to (if
        // possible) have more beams in a given slot.
    }
}

void
NrTestMacSchedulerHarqRrScheduleDlHarq::DoRun()
{
    Config::SetDefault("ns3::NrMacSchedulerHarqRr::ConsolidateHarqRetx", BooleanValue(true));

    // Prepare common settings for both TDMA and OFDMA schedulers
    auto cellConfig = NrMacCschedSapProvider::CschedCellConfigReqParameters();
    cellConfig.m_dlBandwidth = 10; // 10 RBGs
    cellConfig.m_ulBandwidth = 10;

    std::vector<NrMacCschedSapProvider::CschedUeConfigReqParameters> ueConfig;
    for (const auto& dci : m_dcis)
    {
        NrMacCschedSapProvider::CschedUeConfigReqParameters config{};
        config.m_rnti = dci.m_rnti;
        config.m_transmissionMode = 0;
        config.m_beamId = BeamId(dci.m_rnti / 5, 0);
        ueConfig.push_back(config);
    }

    auto* schedSapUser = new TestSchedSapUserHarq([this](auto params) { CheckSchedule(params); },
                                                  [this]() { return m_numSymbols; });
    auto* cschedSapUser = new TestCschedSapUserHarq();

    Ptr<NrMacSchedulerNs3> sched;

    // Instead of using of reshaping straight from scheduler,
    // reproduce the conditions to call it via the scheduler->HARQ scheduler->reshape
    for (bool isTdma : {true, false})
    {
        // Create scheduler
        if (isTdma)
        {
            sched = CreateObject<NrMacSchedulerTdmaRR>();
        }
        else
        {
            sched = CreateObject<NrMacSchedulerOfdmaRR>();
        }
        sched->InstallDlAmc(CreateObject<NrAmc>());
        sched->InstallUlAmc(CreateObject<NrAmc>());

        // Configure scheduler
        sched->SetMacSchedSapUser(schedSapUser);
        sched->SetMacCschedSapUser(cschedSapUser);
        sched->DoCschedCellConfigReq(cellConfig);
        for (const auto& ueConf : ueConfig)
        {
            sched->DoCschedUeConfigReq(ueConf);
        }

        // Set starting symbol
        sched->SetDlCtrlSyms(m_startingSymbol);

        // Create scheduler parameters
        NrMacSchedSapProvider::SchedDlTriggerReqParameters paramsDlTrigger;
        paramsDlTrigger.m_snfSf = SfnSf(0, 0, 0, 0);
        paramsDlTrigger.m_slotType = LteNrTddSlotType::DL;
        paramsDlTrigger.m_dlHarqInfoList = {};

        // Activate harq processes and populate harq info list
        for (const auto& dci : m_dcis)
        {
            auto& ueInfo = sched->m_ueMap.find(dci.m_rnti)->second;
            auto& harqProcess = ueInfo->m_dlHarq.Find(dci.m_rnti)->second;
            harqProcess.m_dciElement = std::make_shared<DciInfoElementTdma>(dci);
            harqProcess.m_active = true;

            DlHarqInfo harqInfo;
            harqInfo.m_harqStatus = DlHarqInfo::NACK;
            harqInfo.m_numRetx = 0;
            harqInfo.m_rnti = dci.m_rnti;
            harqInfo.m_harqProcessId = dci.m_rnti;
            harqInfo.m_bwpIndex = 0;
            paramsDlTrigger.m_dlHarqInfoList.push_back(harqInfo);
        }

        // Indicate CheckSchedule should check for TDMA or OFDMA
        m_testingTdma = isTdma;

        // Call ScheduleDl
        sched->DoSchedDlTriggerReq(paramsDlTrigger);
    }
    delete schedSapUser;
    delete cschedSapUser;
}

/**
 * @brief Regression test for the HARQ round-robin beam ordering
 *
 * NrMacSchedulerHarqRr keeps a persistent round-robin queue of every beam ever
 * seen (m_rrBeams). A previous implementation of GetBeamOrderRR() sized the
 * returned vector to the number of currently active beams while indexing it with
 * the (potentially much larger) size of the persistent queue, causing a heap
 * buffer overflow once more beams had been observed than were active in the
 * current slot. The overflow corrupted the heap and manifested later as a
 * "double free or corruption" abort.
 *
 * This test reuses a single scheduler across several DL HARQ scheduling rounds
 * with a shrinking set of active beams, so that the persistent queue grows
 * larger than the active-beam set, and asserts that scheduling completes without
 * memory corruption.
 */
class NrTestMacSchedulerHarqRrBeamOrder : public TestCase
{
  public:
    NrTestMacSchedulerHarqRrBeamOrder()
        : TestCase("HARQ RR beam order does not overflow with shrinking active beams")
    {
    }

  protected:
    void DoRun() override;

  private:
    /**
     * @brief Drive a single DL HARQ scheduling round on the given scheduler.
     * @param sched Scheduler under test (reused across rounds)
     * @param activeRntis RNTIs (one per beam) whose HARQ process is NACKed in
     *                    this round
     */
    void ScheduleRound(Ptr<NrMacSchedulerNs3> sched, const std::vector<uint16_t>& activeRntis);
};

void
NrTestMacSchedulerHarqRrBeamOrder::ScheduleRound(Ptr<NrMacSchedulerNs3> sched,
                                                 const std::vector<uint16_t>& activeRntis)
{
    NrMacSchedSapProvider::SchedDlTriggerReqParameters paramsDlTrigger;
    paramsDlTrigger.m_snfSf = SfnSf(0, 0, 0, 0);
    paramsDlTrigger.m_slotType = LteNrTddSlotType::DL;
    paramsDlTrigger.m_dlHarqInfoList = {};

    for (auto rnti : activeRntis)
    {
        auto& ueInfo = sched->m_ueMap.find(rnti)->second;
        auto& harqProcess = ueInfo->m_dlHarq.Find(rnti)->second;

        DciInfoElementTdma dci(rnti,
                               DciInfoElementTdma::DL,
                               0,
                               1,
                               10,
                               1,
                               {},
                               100,
                               0,
                               0,
                               DciInfoElementTdma::DATA,
                               0,
                               0);
        dci.m_harqProcess = static_cast<uint8_t>(rnti);
        dci.m_rbgBitmask = std::vector<bool>(10, false);
        dci.m_rbgBitmask.at(0) = true;
        harqProcess.m_dciElement = std::make_shared<DciInfoElementTdma>(dci);
        harqProcess.m_active = true;
        harqProcess.m_status = HarqProcess::WAITING_FEEDBACK;

        DlHarqInfo harqInfo;
        harqInfo.m_harqStatus = DlHarqInfo::NACK;
        harqInfo.m_numRetx = 0;
        harqInfo.m_rnti = rnti;
        harqInfo.m_harqProcessId = static_cast<uint8_t>(rnti);
        harqInfo.m_bwpIndex = 0;
        paramsDlTrigger.m_dlHarqInfoList.push_back(harqInfo);
    }

    sched->DoSchedDlTriggerReq(paramsDlTrigger);
}

void
NrTestMacSchedulerHarqRrBeamOrder::DoRun()
{
    auto cellConfig = NrMacCschedSapProvider::CschedCellConfigReqParameters();
    cellConfig.m_dlBandwidth = 10; // 10 RBGs
    cellConfig.m_ulBandwidth = 10;

    // Configure one UE per beam, with as many beams as RNTIs.
    std::vector<NrMacCschedSapProvider::CschedUeConfigReqParameters> ueConfig;
    const uint16_t numUes = 8;
    for (uint16_t rnti = 0; rnti < numUes; ++rnti)
    {
        NrMacCschedSapProvider::CschedUeConfigReqParameters config{};
        config.m_rnti = rnti;
        config.m_transmissionMode = 0;
        config.m_beamId = BeamId(rnti, 0); // distinct beam per UE
        ueConfig.push_back(config);
    }

    auto* schedSapUser = new TestSchedSapUserHarq([](auto params) {}, []() { return 14; });
    auto* cschedSapUser = new TestCschedSapUserHarq();

    auto sched = CreateObject<NrMacSchedulerTdmaRR>();
    sched->InstallDlAmc(CreateObject<NrAmc>());
    sched->InstallUlAmc(CreateObject<NrAmc>());
    sched->SetMacSchedSapUser(schedSapUser);
    sched->SetMacCschedSapUser(cschedSapUser);
    sched->DoCschedCellConfigReq(cellConfig);
    for (const auto& ueConf : ueConfig)
    {
        sched->DoCschedUeConfigReq(ueConf);
    }
    sched->SetDlCtrlSyms(0);

    // First round: all beams active, so the persistent round-robin queue learns
    // every beam.
    std::vector<uint16_t> allRntis;
    allRntis.reserve(numUes);
    for (uint16_t rnti = 0; rnti < numUes; ++rnti)
    {
        allRntis.push_back(rnti);
    }
    ScheduleRound(sched, allRntis);

    // Subsequent rounds: progressively fewer active beams. With the old code,
    // the result vector (sized to the active-beam count) was indexed by the
    // larger persistent-queue size, overflowing the heap. Reaching this point
    // without an abort confirms the fix.
    for (uint16_t active = numUes - 1; active >= 1 && active < numUes; --active)
    {
        std::vector<uint16_t> rntis;
        rntis.reserve(active);
        for (uint16_t rnti = 0; rnti < active; ++rnti)
        {
            rntis.push_back(rnti);
        }
        ScheduleRound(sched, rntis);
    }

    NS_TEST_EXPECT_MSG_EQ(true, true, "HARQ RR beam ordering completed without memory corruption");

    delete schedSapUser;
    delete cschedSapUser;
}

/**
 * @brief Regression test for the DL HARQ symbol-budget (symAvail) underflow.
 *
 * With ConsolidateHarqRetx disabled (the default), ScheduleDlHarq() used to
 * debit its uint8_t symbol budget (symAvail) twice for the same symbols: once
 * per allocated retransmission, and again at the end of each beam by the beam's
 * symbol span. When a beam's retransmission spanned more than half of the
 * remaining symbols, the second, unguarded subtraction underflowed the budget
 * and wrapped to ~254. The next beam's retransmissions then passed every check
 * and were placed past the end of the slot, so ScheduleDlHarq() returned a span
 * larger than the budget it was given and the caller tripped
 * NS_ASSERT_MSG(dlSymAvail >= usedHarq, ...).
 *
 * This test places one retransmission per beam in two distinct beams, each
 * spanning more than half of the slot, and checks that no DCI is scheduled past
 * the end of the slot (and, in a build with assertions enabled, that the caller
 * assertion is not tripped). It is exercised on both the TDMA and OFDMA
 * round-robin schedulers, since both share NrMacSchedulerHarqRr.
 */
class NrTestMacSchedulerHarqRrSymbolBudget : public TestCase
{
  public:
    /**
     * @brief Constructor
     * @param isTdma whether to test the TDMA (true) or OFDMA (false) scheduler
     */
    NrTestMacSchedulerHarqRrSymbolBudget(bool isTdma)
        : TestCase(std::string("HARQ DL symbol budget does not underflow across beams (") +
                   (isTdma ? "TDMA)" : "OFDMA)")),
          m_isTdma(isTdma)
    {
    }

  protected:
    void DoRun() override;

  private:
    /**
     * @brief Verify that no scheduled HARQ DCI extends past the end of the slot.
     * @param params the scheduled slot allocation reported by the scheduler
     */
    void CheckSchedule(const NrMacSchedSapUser::SchedConfigIndParameters& params);

    bool m_isTdma;                                 //!< Test TDMA (true) or OFDMA (false)
    static constexpr uint8_t m_symbolsPerSlot{14}; //!< Symbols available in the slot
};

void
NrTestMacSchedulerHarqRrSymbolBudget::CheckSchedule(
    const NrMacSchedSapUser::SchedConfigIndParameters& params)
{
    uint16_t lastUsedSymbol = 0;
    for (const auto& varTtiAllocInfo : params.m_slotAllocInfo.m_varTtiAllocInfo)
    {
        // Only data DCIs carry HARQ retransmissions
        if (varTtiAllocInfo.m_dci->m_type != DciInfoElementTdma::DATA)
        {
            continue;
        }
        lastUsedSymbol =
            std::max<uint16_t>(lastUsedSymbol,
                               varTtiAllocInfo.m_dci->m_symStart + varTtiAllocInfo.m_dci->m_numSym);
    }
    NS_TEST_ASSERT_MSG_LT_OR_EQ(
        +lastUsedSymbol,
        +m_symbolsPerSlot,
        (m_isTdma ? "TDMA" : "OFDMA")
            << ": HARQ retransmissions were placed past the end of the slot "
               "(symAvail underflowed and over-allocated symbols)");
}

void
NrTestMacSchedulerHarqRrSymbolBudget::DoRun()
{
    // The bug lived in the default, non-consolidating path. Set the attribute
    // explicitly so the test does not depend on the default value (and is robust
    // to other test cases changing it via Config::SetDefault).
    Config::SetDefault("ns3::NrMacSchedulerHarqRr::ConsolidateHarqRetx", BooleanValue(false));

    auto cellConfig = NrMacCschedSapProvider::CschedCellConfigReqParameters();
    cellConfig.m_dlBandwidth = 10; // 10 RBGs
    cellConfig.m_ulBandwidth = 10;

    // One UE per beam, in two distinct beams. Each retransmission spans 8 of the
    // 14 available symbols: more than half, so a single beam leaves 6 symbols and
    // the buggy double-subtraction wraps the budget below zero, letting the
    // second beam over-allocate.
    const uint8_t numSym = 8;
    const std::vector<uint16_t> rntis = {0, 1};

    std::vector<NrMacCschedSapProvider::CschedUeConfigReqParameters> ueConfig;
    for (auto rnti : rntis)
    {
        NrMacCschedSapProvider::CschedUeConfigReqParameters config{};
        config.m_rnti = rnti;
        config.m_transmissionMode = 0;
        config.m_beamId = BeamId(rnti, 0); // distinct beam per UE
        ueConfig.push_back(config);
    }

    auto* schedSapUser = new TestSchedSapUserHarq([this](auto params) { CheckSchedule(params); },
                                                  []() { return uint32_t{m_symbolsPerSlot}; });
    auto* cschedSapUser = new TestCschedSapUserHarq();

    Ptr<NrMacSchedulerNs3> sched;
    if (m_isTdma)
    {
        sched = CreateObject<NrMacSchedulerTdmaRR>();
    }
    else
    {
        sched = CreateObject<NrMacSchedulerOfdmaRR>();
    }
    sched->InstallDlAmc(CreateObject<NrAmc>());
    sched->InstallUlAmc(CreateObject<NrAmc>());
    sched->SetMacSchedSapUser(schedSapUser);
    sched->SetMacCschedSapUser(cschedSapUser);
    sched->DoCschedCellConfigReq(cellConfig);
    for (const auto& ueConf : ueConfig)
    {
        sched->DoCschedUeConfigReq(ueConf);
    }
    sched->SetDlCtrlSyms(0);

    // Build a NACKed HARQ process per UE so ScheduleDlHarq retransmits them.
    NrMacSchedSapProvider::SchedDlTriggerReqParameters paramsDlTrigger;
    paramsDlTrigger.m_snfSf = SfnSf(0, 0, 0, 0);
    paramsDlTrigger.m_slotType = LteNrTddSlotType::DL;
    for (auto rnti : rntis)
    {
        auto& ueInfo = sched->m_ueMap.find(rnti)->second;
        auto& harqProcess = ueInfo->m_dlHarq.Find(rnti)->second;

        DciInfoElementTdma dci(rnti,
                               DciInfoElementTdma::DL,
                               0,
                               numSym,
                               10,
                               1,
                               {},
                               100,
                               0,
                               0,
                               DciInfoElementTdma::DATA,
                               0,
                               0);
        dci.m_harqProcess = static_cast<uint8_t>(rnti);
        dci.m_rbgBitmask = std::vector<bool>(10, true); // full-band allocation
        harqProcess.m_dciElement = std::make_shared<DciInfoElementTdma>(dci);
        harqProcess.m_active = true;
        harqProcess.m_status = HarqProcess::WAITING_FEEDBACK;

        DlHarqInfo harqInfo;
        harqInfo.m_harqStatus = DlHarqInfo::NACK;
        harqInfo.m_numRetx = 0;
        harqInfo.m_rnti = rnti;
        harqInfo.m_harqProcessId = static_cast<uint8_t>(rnti);
        harqInfo.m_bwpIndex = 0;
        paramsDlTrigger.m_dlHarqInfoList.push_back(harqInfo);
    }

    // With the bug present this either trips an assertion (debug build) or
    // schedules a retransmission past the end of the slot, caught by
    // CheckSchedule above.
    sched->DoSchedDlTriggerReq(paramsDlTrigger);

    delete schedSapUser;
    delete cschedSapUser;
}

/**
 * @brief Verify OFDMA HARQ retransmissions in the same beam share symbols.
 *
 * In OFDMA, several UEs in the same beam are retransmitted on the same OFDM
 * symbols but disjoint RBGs (frequency multiplexing). ScheduleDlHarq() must
 * therefore debit the slot symbol budget by the beam's symbol span, not by the
 * sum of the per-UE spans. The earlier per-retransmission debit (since fixed)
 * dropped every UE after the first whenever the shared span exceeded half the
 * slot, silently losing retransmissions (and, with a following beam, underflowed
 * the budget). This test schedules two UEs in one beam on disjoint RBGs, each
 * spanning more than half of the slot, and checks that both are scheduled within
 * the slot, with and without HARQ consolidation.
 */
class NrTestMacSchedulerHarqRrOfdmaSharing : public TestCase
{
  public:
    /**
     * @brief Constructor
     * @param consolidate whether ConsolidateHarqRetx (allocation reshaping) is enabled
     */
    NrTestMacSchedulerHarqRrOfdmaSharing(bool consolidate)
        : TestCase(std::string("OFDMA HARQ retransmissions share symbols within a beam (") +
                   (consolidate ? "consolidated)" : "not consolidated)")),
          m_consolidate(consolidate)
    {
    }

  protected:
    void DoRun() override;

  private:
    /**
     * @brief Check that both same-beam UEs were scheduled within the slot.
     * @param params the scheduled slot allocation reported by the scheduler
     */
    void CheckSchedule(const NrMacSchedSapUser::SchedConfigIndParameters& params);

    bool m_consolidate;                            //!< Reshape (consolidate) HARQ retx or not
    bool m_checked{false};                         //!< Set once the scheduler reports a slot
    static constexpr uint8_t m_symbolsPerSlot{14}; //!< Symbols available in the slot
    static constexpr uint8_t m_numSym{8};          //!< Symbols per retransmission (> half the slot)
    static constexpr uint32_t m_numUes{2};         //!< UEs sharing the single beam
};

void
NrTestMacSchedulerHarqRrOfdmaSharing::CheckSchedule(
    const NrMacSchedSapUser::SchedConfigIndParameters& params)
{
    m_checked = true;

    uint32_t numDataDcis = 0;
    uint16_t lastUsedSymbol = 0;
    for (const auto& varTtiAllocInfo : params.m_slotAllocInfo.m_varTtiAllocInfo)
    {
        if (varTtiAllocInfo.m_dci->m_type != DciInfoElementTdma::DATA)
        {
            continue;
        }
        ++numDataDcis;
        lastUsedSymbol =
            std::max<uint16_t>(lastUsedSymbol,
                               varTtiAllocInfo.m_dci->m_symStart + varTtiAllocInfo.m_dci->m_numSym);
    }

    // Both UEs must be retransmitted: since each spans more than half the slot,
    // the only way to fit both is to share the same symbols on disjoint RBGs.
    NS_TEST_ASSERT_MSG_EQ(numDataDcis,
                          m_numUes,
                          "Both same-beam OFDMA UEs should be retransmitted by sharing symbols, "
                          "but some were dropped");
    NS_TEST_ASSERT_MSG_LT_OR_EQ(+lastUsedSymbol,
                                +m_symbolsPerSlot,
                                "HARQ retransmissions were placed past the end of the slot");
}

void
NrTestMacSchedulerHarqRrOfdmaSharing::DoRun()
{
    Config::SetDefault("ns3::NrMacSchedulerHarqRr::ConsolidateHarqRetx",
                       BooleanValue(m_consolidate));

    auto cellConfig = NrMacCschedSapProvider::CschedCellConfigReqParameters();
    cellConfig.m_dlBandwidth = 10; // 10 RBGs
    cellConfig.m_ulBandwidth = 10;

    // Two UEs in the SAME beam, on disjoint RBG halves, each spanning 8 of the 14
    // available symbols. Stacking them in time would need 16 > 14 symbols, so the
    // scheduler can only fit both by sharing the symbols across the two RBG sets.
    const std::vector<uint16_t> rntis = {0, 1};

    std::vector<NrMacCschedSapProvider::CschedUeConfigReqParameters> ueConfig;
    for (auto rnti : rntis)
    {
        NrMacCschedSapProvider::CschedUeConfigReqParameters config{};
        config.m_rnti = rnti;
        config.m_transmissionMode = 0;
        config.m_beamId = BeamId(0, 0); // single shared beam
        ueConfig.push_back(config);
    }

    auto* schedSapUser = new TestSchedSapUserHarq([this](auto params) { CheckSchedule(params); },
                                                  []() { return uint32_t{m_symbolsPerSlot}; });
    auto* cschedSapUser = new TestCschedSapUserHarq();

    auto sched = CreateObject<NrMacSchedulerOfdmaRR>();
    sched->InstallDlAmc(CreateObject<NrAmc>());
    sched->InstallUlAmc(CreateObject<NrAmc>());
    sched->SetMacSchedSapUser(schedSapUser);
    sched->SetMacCschedSapUser(cschedSapUser);
    sched->DoCschedCellConfigReq(cellConfig);
    for (const auto& ueConf : ueConfig)
    {
        sched->DoCschedUeConfigReq(ueConf);
    }
    sched->SetDlCtrlSyms(0);

    NrMacSchedSapProvider::SchedDlTriggerReqParameters paramsDlTrigger;
    paramsDlTrigger.m_snfSf = SfnSf(0, 0, 0, 0);
    paramsDlTrigger.m_slotType = LteNrTddSlotType::DL;
    for (size_t i = 0; i < rntis.size(); ++i)
    {
        const auto rnti = rntis.at(i);
        auto& ueInfo = sched->m_ueMap.find(rnti)->second;
        auto& harqProcess = ueInfo->m_dlHarq.Find(rnti)->second;

        // Disjoint RBG halves: UE 0 -> RBGs [0,4], UE 1 -> RBGs [5,9].
        std::vector<bool> bitmask(10, false);
        for (size_t rbg = i * 5; rbg < (i + 1) * 5; ++rbg)
        {
            bitmask.at(rbg) = true;
        }

        DciInfoElementTdma dci(rnti,
                               DciInfoElementTdma::DL,
                               0,
                               m_numSym,
                               10,
                               1,
                               {},
                               100,
                               0,
                               0,
                               DciInfoElementTdma::DATA,
                               0,
                               0);
        dci.m_harqProcess = static_cast<uint8_t>(rnti);
        dci.m_rbgBitmask = bitmask;
        harqProcess.m_dciElement = std::make_shared<DciInfoElementTdma>(dci);
        harqProcess.m_active = true;
        harqProcess.m_status = HarqProcess::WAITING_FEEDBACK;

        DlHarqInfo harqInfo;
        harqInfo.m_harqStatus = DlHarqInfo::NACK;
        harqInfo.m_numRetx = 0;
        harqInfo.m_rnti = rnti;
        harqInfo.m_harqProcessId = static_cast<uint8_t>(rnti);
        harqInfo.m_bwpIndex = 0;
        paramsDlTrigger.m_dlHarqInfoList.push_back(harqInfo);
    }

    sched->DoSchedDlTriggerReq(paramsDlTrigger);

    NS_TEST_ASSERT_MSG_EQ(m_checked,
                          true,
                          "Scheduler did not report a slot allocation; checks did not run");

    delete schedSapUser;
    delete cschedSapUser;
}

class NrTestSchedHarqSuite : public TestSuite
{
  public:
    NrTestSchedHarqSuite()
        : TestSuite("nr-test-sched-harq", Type::UNIT)
    {
        AddTestCase(new NrTestMacSchedulerHarqRrBeamOrder(), Duration::QUICK);
        AddTestCase(new NrTestMacSchedulerHarqRrSymbolBudget(true /* TDMA */), Duration::QUICK);
        AddTestCase(new NrTestMacSchedulerHarqRrSymbolBudget(false /* OFDMA */), Duration::QUICK);
        AddTestCase(new NrTestMacSchedulerHarqRrOfdmaSharing(false /* not consolidated */),
                    Duration::QUICK);
        AddTestCase(new NrTestMacSchedulerHarqRrOfdmaSharing(true /* consolidated */),
                    Duration::QUICK);

        // clang-format off
        using DIET = DciInfoElementTdma;
        std::vector<DciInfoElementTdma> dcis{
            // beam 0
            // rnti,   format, startSym, numSym, mcs, rank, precmat, tbs, ndi, rv,       type, bwp, tpc
            {     0, DIET::DL,        0,      4,  10,    4,      {}, 800,   0,  0, DIET::DATA,   0,   0},
            {     1, DIET::DL,        1,      7,  17,    1,      {}, 200,   0,  0, DIET::DATA,   0,   0},
            {     2, DIET::DL,        2,      1,  13,    3,      {}, 600,   0,  0, DIET::DATA,   0,   0},
            {     3, DIET::DL,        3,      9,  10,    2,      {}, 400,   0,  0, DIET::DATA,   0,   0},
            {     4, DIET::DL,        4,      2,   6,    4,      {}, 800,   0,  0, DIET::DATA,   0,   0},
            // beam 1
            {     5, DIET::DL,        5,      5,  20,    4,      {}, 800,   0,  0, DIET::DATA,   0,   0},
            {     6, DIET::DL,        6,      3,  10,    1,      {}, 200,   0,  0, DIET::DATA,   0,   0},
            {     7, DIET::DL,        7,      8,   1,    1,      {}, 200,   0,  0, DIET::DATA,   0,   0},
            {     8, DIET::DL,        8,      1,   7,    3,      {}, 600,   0,  0, DIET::DATA,   0,   0},
            {     9, DIET::DL,        9,      2,  19,    2,      {}, 400,   0,  0, DIET::DATA,   0,   0}
        };
        dcis.at(0).m_rbgBitmask = { true,  true, false, false, false,  true, false, false,  true,  true};
        dcis.at(1).m_rbgBitmask = {false,  true,  true, false,  true,  true,  true, false,  true, false};
        dcis.at(2).m_rbgBitmask = {false, false,  true, false, false,  true,  true,  true,  true,  true};
        dcis.at(3).m_rbgBitmask = { true, false,  true, false,  true, false,  true, false,  true, false};
        dcis.at(4).m_rbgBitmask = {false,  true, false,  true, false,  true, false,  true, false,  true};
        dcis.at(5).m_rbgBitmask = { true, false,  true,  true, false, false, false, false, false,  true};
        dcis.at(6).m_rbgBitmask = { true, false, false,  true, false,  true, false, false, false,  true};
        dcis.at(7).m_rbgBitmask = { true, false, false, false,  true,  true,  true, false, false,  true};
        dcis.at(8).m_rbgBitmask = { true, false, false, false, false,  true,  true,  true, false,  true};
        dcis.at(9).m_rbgBitmask = { true,  true, false,  true, false,  true,  true, false, false, false};
        dcis.at(0).m_harqProcess = 0;
        dcis.at(1).m_harqProcess = 1;
        dcis.at(2).m_harqProcess = 2;
        dcis.at(3).m_harqProcess = 3;
        dcis.at(4).m_harqProcess = 4;
        dcis.at(5).m_harqProcess = 5;
        dcis.at(6).m_harqProcess = 6;
        dcis.at(7).m_harqProcess = 7;
        dcis.at(8).m_harqProcess = 8;
        dcis.at(9).m_harqProcess = 9;
        // clang-format on
        for (auto [startSym, numSym] : std::vector<std::pair<uint8_t, uint8_t>>{
                 {0, 0},
                 {0, 13},
                 {6, 14},
                 {0, 1},
                 {1, 13},
             })
        {
            // Test reshaping alone
            AddTestCase(new NrTestMacSchedulerHarqRrReshape(
                            std::vector<DIET>(dcis.begin(), dcis.begin() + 4),
                            startSym,
                            numSym,
                            "Reshape: Beam   0, startSym " + std::to_string(startSym) +
                                ", numSym " + std::to_string(numSym)),
                        Duration::QUICK);
            AddTestCase(new NrTestMacSchedulerHarqRrReshape(
                            std::vector<DIET>(dcis.begin() + 6, dcis.end()),
                            startSym,
                            numSym,
                            "Reshape: Beam   1, startSym " + std::to_string(startSym) +
                                ", numSym " + std::to_string(numSym)),
                        Duration::QUICK);
            AddTestCase(new NrTestMacSchedulerHarqRrReshape(
                            std::vector<DIET>(dcis.begin() + 3, dcis.begin() + 7),
                            startSym,
                            numSym,
                            "Reshape: Beam 0+1, startSym " + std::to_string(startSym) +
                                ", numSym " + std::to_string(numSym)),
                        Duration::QUICK);
            // Test reshaping via scheduler
            AddTestCase(new NrTestMacSchedulerHarqRrScheduleDlHarq(
                            std::vector<DIET>(dcis.begin(), dcis.begin() + 1),
                            startSym,
                            numSym,
                            "Reshape with scheduler: Beam   0, startSym " +
                                std::to_string(startSym) + ", numSym " + std::to_string(numSym)),
                        Duration::QUICK);
            AddTestCase(new NrTestMacSchedulerHarqRrScheduleDlHarq(
                            std::vector<DIET>(dcis.begin() + 1, dcis.begin() + 2),
                            startSym,
                            numSym,
                            "Reshape with scheduler: Beam   0, startSym " +
                                std::to_string(startSym) + ", numSym " + std::to_string(numSym)),
                        Duration::QUICK);
            AddTestCase(new NrTestMacSchedulerHarqRrScheduleDlHarq(
                            std::vector<DIET>(dcis.begin() + 2, dcis.begin() + 3),
                            startSym,
                            numSym,
                            "Reshape with scheduler: Beam   0, startSym " +
                                std::to_string(startSym) + ", numSym " + std::to_string(numSym)),
                        Duration::QUICK);
            AddTestCase(new NrTestMacSchedulerHarqRrScheduleDlHarq(
                            std::vector<DIET>(dcis.begin() + 3, dcis.begin() + 4),
                            startSym,
                            numSym,
                            "Reshape with scheduler: Beam   0, startSym " +
                                std::to_string(startSym) + ", numSym " + std::to_string(numSym)),
                        Duration::QUICK);
            AddTestCase(new NrTestMacSchedulerHarqRrScheduleDlHarq(
                            std::vector<DIET>(dcis.begin(), dcis.begin() + 4),
                            startSym,
                            numSym,
                            "Reshape with scheduler: Beam   0, startSym " +
                                std::to_string(startSym) + ", numSym " + std::to_string(numSym)),
                        Duration::QUICK);
            AddTestCase(new NrTestMacSchedulerHarqRrScheduleDlHarq(
                            std::vector<DIET>(dcis.begin() + 4, dcis.begin() + 6),
                            startSym,
                            numSym,
                            "Reshape with scheduler: Beam 0+1, startSym " +
                                std::to_string(startSym) + ", numSym " + std::to_string(numSym)),
                        Duration::QUICK);
            AddTestCase(new NrTestMacSchedulerHarqRrScheduleDlHarq(
                            std::vector<DIET>(dcis.begin() + 3, dcis.begin() + 7),
                            startSym,
                            numSym,
                            "Reshape with scheduler: Beam 0+1, startSym " +
                                std::to_string(startSym) + ", numSym " + std::to_string(numSym)),
                        Duration::QUICK);
            AddTestCase(new NrTestMacSchedulerHarqRrScheduleDlHarq(
                            std::vector<DIET>(dcis.begin() + 2, dcis.begin() + 8),
                            startSym,
                            numSym,
                            "Reshape with scheduler: Beam 0+1, startSym " +
                                std::to_string(startSym) + ", numSym " + std::to_string(numSym)),
                        Duration::QUICK);
        }
    }
};

static NrTestSchedHarqSuite nrSchedHarqTestSuite; //!< NR HARQ scheduler test suite

} // namespace ns3
