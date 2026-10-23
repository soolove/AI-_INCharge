// Copyright (c) 2026 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
//
// SPDX-License-Identifier: GPL-2.0-only

/**
 * @ingroup test
 * @file nr-test-rlc-um-rx.cc
 *
 * @brief Test suite `nr-test-rlc-um-rx`: white-box regression tests of the NrRlcUm reception and
 * transmission buffers, driving an NrRlcUm instance wired to test PDCP/MAC entities with
 * hand-crafted PDUs and pre-set state variables. The cases verify that buffered SNs are
 * reassembled (and removed) before VR(UR) advances so no PDU is left outside the reception
 * window, including at 10-bit SN wrap-around; that a stranded partial SDU is discarded on
 * t-Reordering expiry so no corrupt SDU is delivered to PDCP; that an SDU discarded by the PDCP
 * discard timer fires the TxDrop trace once and is not enqueued in the Tx buffer; and that
 * ReassembleSnInterval skips missing SNs safely with logging enabled while reassembling and
 * delivering the SNs that are present.
 */

#include "nr-test-entities.h"

#include "ns3/boolean.h"
#include "ns3/log.h"
#include "ns3/nr-rlc-header.h"
#include "ns3/nr-rlc-tag.h"
#include "ns3/nr-rlc-um.h"
#include "ns3/packet.h"
#include "ns3/simulator.h"
#include "ns3/uinteger.h"

NS_LOG_COMPONENT_DEFINE("NrRlcUmTestCase");

namespace ns3
{

class NrRlcUmTestCase : public TestCase
{
  public:
    NrRlcUmTestCase()
        : TestCase("Test RLC UM RX: Check packets not reassembled when window advances")
    {
    }

  private:
    void DoRun() override;

    /**
     * Create an RLC PDU
     *
     * @param payloadSize Size of the RLC PDU payload in bytes
     * @param rlcSn RLC Sequence Number (SN) of the PDU
     * @param extensionBit Extension Bit flag:
     *                     0 = no more Length Indicators (LI) follow;
     *                     1 = at least one more LI follows
     *                     Default: DATA_FIELD_FOLLOWS (LI = 0)
     * @param framingInfo Framing Info (FI) flag, indicates SDU boundaries:
     *                    00 = starts and ends with an SDU
     *                    01 = starts with an SDU but does not end with one
     *                    10 = does not start with an SDU but ends with one
     *                    11 = neither starts nor ends with an SDU
     *                    Default: FIRST_BYTE | LAST_BYTE (FI = 00)
     * @return A Ptr<Packet> representing the RLC PDU
     */
    Ptr<Packet> CreateRlcPdu(uint32_t payloadSize,
                             uint16_t rlcSn,
                             uint8_t extensionBit = NrRlcHeader::DATA_FIELD_FOLLOWS,
                             uint8_t framingInfo = NrRlcHeader::FIRST_BYTE |
                                                   NrRlcHeader::LAST_BYTE);

    Ptr<NrTestPdcp> rxPdcp; ///< The reception PDCP
};

Ptr<Packet>
NrRlcUmTestCase::CreateRlcPdu(uint32_t payloadSize,
                              uint16_t rlcSn,
                              uint8_t extensionBit,
                              uint8_t framingInfo)
{
    // Create dummy payload of given size
    Ptr<Packet> rlcPdu = Create<Packet>(payloadSize);

    // Create RLC header
    NrRlcHeader rlcHeader;
    rlcHeader.SetSequenceNumber(nr::SequenceNumber10(rlcSn));
    rlcHeader.PushExtensionBit(extensionBit);
    rlcHeader.SetFramingInfo(framingInfo);

    // Add header to packet
    rlcPdu->AddHeader(rlcHeader);

    return rlcPdu;
}

void
NrRlcUmTestCase::DoRun()
{
    auto logLevel = (LogLevel)(LOG_PREFIX_FUNC | LOG_PREFIX_TIME | LOG_LEVEL_ALL);
    LogComponentEnable("NrRlcUmTestCase", logLevel);
    LogComponentEnable("NrRlcUm", logLevel);

    NS_LOG_INFO("-----------------------------------------------------------------------------");
    NS_LOG_INFO("DoRun started");

    // Instantiate RLC
    Ptr<NrRlcUm> rlc = CreateObject<NrRlcUm>();

    // Create transmission PDCP test entity
    rxPdcp = CreateObject<NrTestPdcp>();

    // Connect RLC with Test PDCP
    rlc->SetNrRlcSapUser(rxPdcp->GetNrRlcSapUser());

    // 1) Bug check: In the implementation before MR !354, SNs {1016, 1017} remained in m_rxBuffer
    // un-reassembled. In the updated implementation (MR !354), these PDUs are reassembled before
    // advancing VR(UR) to 1018.

    NS_LOG_INFO("Step 1.1: Initialized m_rxBuffer with SN 0, 502, 1016, 1017, 1018");
    rlc->m_rxBuffer[0] = CreateRlcPdu(10, 0, 0x00, NrRlcHeader::FIRST_BYTE);
    rlc->m_rxBuffer[1016] = CreateRlcPdu(20, 1016, 0x00, NrRlcHeader::FIRST_BYTE);
    rlc->m_rxBuffer[1017] =
        CreateRlcPdu(15, 1017, 0x00, NrRlcHeader::FIRST_BYTE | NrRlcHeader::NO_LAST_BYTE);
    rlc->m_rxBuffer[1018] =
        CreateRlcPdu(15, 1018, 0x00, NrRlcHeader::NO_FIRST_BYTE | NrRlcHeader::LAST_BYTE);
    rlc->m_rxBuffer[502] = CreateRlcPdu(30, 502, 0x00, NrRlcHeader::FIRST_BYTE);

    NS_LOG_INFO(
        "Step 1.2: Reception window set (VR(UR)=1015, VR(UH)=503, windowSize=512) -> [1015, 503)");
    rlc->m_vrUr = 1015;
    rlc->m_vrUx = 0;
    rlc->m_vrUh = 503;
    rlc->m_windowSize = 512;

    rlc->m_expectedSeqNumber = 1015;
    rlc->m_reassemblingState = NrRlcUm::WAITING_S0_FULL;

    // Create RLC PDU with SN=505, FI = 00, E = 0
    NS_LOG_INFO("Step 1.3: Creating PDU with SN=505");
    Ptr<Packet> pdu = CreateRlcPdu(50, 505, 0x00, NrRlcHeader::FIRST_BYTE);

    // Add RLC packet tag
    NrRlcTag rlcTag(Simulator::Now());
    pdu->AddByteTag(rlcTag);

    NrMacSapUser::ReceivePduParameters rxPduParams = {};
    rxPduParams.p = pdu;
    rxPduParams.lcid = 0;
    rxPduParams.rnti = 0;

    NS_LOG_INFO("Step 1.4: Calling DoReceivePdu with SN=505");
    NS_LOG_INFO("-----------------------------------------------------------------------------");
    rlc->DoReceivePdu(rxPduParams);

    NS_LOG_INFO("-----------------------------------------------------------------------------");
    NS_LOG_INFO("Step 1.5: DoReceivePdu finished");

    for (const auto& entry : rlc->m_rxBuffer)
    {
        uint16_t sn = entry.first;
        uint16_t vrUr = rlc->m_vrUr.GetValue();
        uint16_t vrUh = rlc->m_vrUh.GetValue();

        bool insideWindow;

        if (vrUr <= vrUh)
        {
            // No wrap
            insideWindow = (sn >= vrUr && sn < vrUh);
        }
        else
        {
            // Wrap-around
            insideWindow = (sn >= vrUr || sn < vrUh);
        }

        if (!insideWindow)
        {
            NS_LOG_ERROR(
                " ERROR: Packet stored outside reception window, SN=" + std::to_string(sn) +
                ", [VR(UR)=" + std::to_string(vrUr) + ", VR(UH)=" + std::to_string(vrUh) + ")");

            // Trigger the assert
            NS_TEST_ASSERT_MSG_EQ(false, true, "");
        }
    }

    NS_LOG_INFO("m_rxBuffer key (SN) contents after DoReceivePdu:");
    for (const auto& entry : rlc->m_rxBuffer)
    {
        NS_LOG_INFO("SN = " << entry.first);
    }

    NS_LOG_INFO("-----------------------------------------------------------------------------");

    // 2) Bug check: In the implementation before MR !354, SN {1016, 1017} remain in m_rxBuffer
    // without being reassembled. Therefore, when a new PDU with SN=1016 is received by RLC UM RX,
    // the code detects that a PDU with the same SN already exists in the buffer and the ASSERT is
    // triggered. In the updated implementation (MR !354), these PDUs are reassembled before
    // advancing VR(UR) to 1018, so they are removed from m_rxBuffer.

    NS_LOG_INFO(
        "Step 2.1: Reception window set (VR(UR)=504, VR(UH)=1016, windowSize=512) -> [504, 1016)");
    rlc->m_vrUr = 504;
    rlc->m_vrUx = 0;
    rlc->m_vrUh = 1016;
    rlc->m_windowSize = 512;

    rlc->m_expectedSeqNumber = 504;
    rlc->m_reassemblingState = NrRlcUm::WAITING_S0_FULL;

    NS_LOG_INFO("Step 2.2: Store SN=1015 and remove SN=502 from the RX buffer (m_rxBuffer), as it "
                "should have already been reassembled");
    rlc->m_rxBuffer[1015] = CreateRlcPdu(20, 1015, 0x00, NrRlcHeader::FIRST_BYTE);
    rlc->m_rxBuffer.erase(502);

    NS_LOG_INFO("Step 2.3: Creating PDU with SN=1016");
    Ptr<Packet> pdu_1016 =
        CreateRlcPdu(50, 1016, 0x00, NrRlcHeader::FIRST_BYTE | NrRlcHeader::NO_LAST_BYTE);
    // Add RLC packet tag
    NrRlcTag rlcTag_1016(Simulator::Now());
    pdu_1016->AddByteTag(rlcTag_1016);
    rxPduParams.p = pdu_1016;

    NS_LOG_INFO("Step 2.4: Calling DoReceivePdu with SN=1016");
    NS_LOG_INFO("-----------------------------------------------------------------------------");
    rlc->DoReceivePdu(rxPduParams);
    NS_LOG_INFO("-----------------------------------------------------------------------------");
    NS_LOG_INFO("m_rxBuffer key (SN) contents after DoReceivePdu:");
    for (const auto& entry : rlc->m_rxBuffer)
    {
        NS_LOG_INFO("SN = " << entry.first);
    }
    NS_LOG_INFO("-----------------------------------------------------------------------------");

    // 3) Bug check: In the implementation before MR !354, some PDUs were not removed from the RX
    // buffer. This could cause the same SN to be received again in a new cycle while a PDU with the
    // same SN from the previous cycle still exists in m_rxBuffer. For example, SN=1016 could
    // arrive again even though SN=1016 from the previous cycle is still present. If the current
    // PDU with SN=1016 is lost (corrupted or not received), the leftover PDU from the previous
    // cycle could be processed, leading to an incorrect FI transition.
    //
    // In the updated implementation (MR !354), these PDUs are reassembled before advancing VR(UR)
    // to 1018, ensuring they are removed from m_rxBuffer and avoiding such incorrect FI
    // transitions.

    NS_LOG_INFO(
        "Step 3.1: Reception window set (VR(UR)=1014, VR(UH)=502, windowSize=512) -> [1014, 502)");
    rlc->m_vrUr = 1014;
    rlc->m_vrUx = 0;
    rlc->m_vrUh = 502;
    rlc->m_windowSize = 512;

    rlc->m_expectedSeqNumber = 1014;
    rlc->m_reassemblingState = NrRlcUm::WAITING_S0_FULL;

    NS_LOG_INFO("Step 3.2: Creating PDU with SN=502");
    Ptr<Packet> pdu_502 =
        CreateRlcPdu(50, 502, 0x00, NrRlcHeader::FIRST_BYTE | NrRlcHeader::NO_LAST_BYTE);
    // Add RLC packet tag
    NrRlcTag rlcTag_502(Simulator::Now());
    pdu_502->AddByteTag(rlcTag_502);
    rxPduParams.p = pdu_502;

    NS_LOG_INFO("Step 3.3: Calling DoReceivePdu with SN=502");
    NS_LOG_INFO("-----------------------------------------------------------------------------");
    rlc->DoReceivePdu(rxPduParams);

    NS_LOG_INFO("-----------------------------------------------------------------------------");
    NS_LOG_INFO("m_rxBuffer key (SN) contents after DoReceivePdu:");
    for (const auto& entry : rlc->m_rxBuffer)
    {
        NS_LOG_INFO("SN = " << entry.first);
    }
    NS_LOG_INFO("-----------------------------------------------------------------------------");

    // 4) Check wrap-around: Set the window VR(UR) to 1023 with a packet SN=0 already in the buffer,
    // and verify that the wrap-around behaves correctly.

    NS_LOG_INFO("Step 4.1: Window check, Reception window set (VR(UR)=1023, VR(UH)=511, "
                "windowSize=512) -> [1023, 511)");
    rlc->m_vrUr = 1023;
    rlc->m_vrUx = 0;
    rlc->m_vrUh = 511;
    rlc->m_windowSize = 512;

    rlc->m_expectedSeqNumber = 1023;
    rlc->m_reassemblingState = NrRlcUm::WAITING_S0_FULL;

    NS_LOG_INFO("Step 4.2: Creating PDU with SN=511");
    Ptr<Packet> pdu_511 =
        CreateRlcPdu(50, 511, 0x00, NrRlcHeader::FIRST_BYTE | NrRlcHeader::NO_LAST_BYTE);
    // Add RLC packet tag
    NrRlcTag rlcTag_511(Simulator::Now());
    pdu_511->AddByteTag(rlcTag_511);
    rxPduParams.p = pdu_511;

    NS_LOG_INFO("Step 4.3: Calling DoReceivePdu with SN=511");
    NS_LOG_INFO("-----------------------------------------------------------------------------");
    rlc->DoReceivePdu(rxPduParams);

    // Flush the event scheduled by the NrTestPdcp constructor, so that it does not
    // outlive this test case and fire on a destroyed object in the next one.
    Simulator::Destroy();
}

/**
 * @ingroup tests
 *
 * @brief Regression test for the RLC UM reassembly t-Reordering discard.
 *
 * Reproduces the condition that produced corrupt (non-IP) SDUs during fast handovers:
 * a first SDU segment (S0) is received and kept (state WAITING_SI_SF) awaiting a
 * continuation that is then lost. Per TS 38.322 (5.1.2.2.4 / 5.2.2.2) the un-completable
 * partial SDU must be discarded when t-Reordering expires. Without that discard the stale
 * S0 survives until the 10-bit SN wraps (~1024 PDUs), after which an unrelated later
 * segment aliases as the awaited continuation and is concatenated onto the stale head,
 * delivering a corrupt SDU upwards. This test asserts the discard happens.
 */
class NrRlcUmReorderingDiscardTestCase : public TestCase
{
  public:
    NrRlcUmReorderingDiscardTestCase()
        : TestCase("Test RLC UM RX: stranded partial SDU discarded on t-Reordering expiry")
    {
    }

  private:
    void DoRun() override;
};

void
NrRlcUmReorderingDiscardTestCase::DoRun()
{
    Ptr<NrRlcUm> rlc = CreateObject<NrRlcUm>();
    Ptr<NrTestPdcp> rxPdcp = CreateObject<NrTestPdcp>();
    rlc->SetNrRlcSapUser(rxPdcp->GetNrRlcSapUser());

    // Put the reassembler mid-SDU: a first segment (S0) has been received and is held,
    // awaiting its continuation at SN = m_expectedSeqNumber (= VR(UR) = 200). That
    // continuation is lost (not in m_rxBuffer) and a reordering gap is open.
    rlc->m_keepS0 = Create<Packet>(40);
    rlc->m_reassemblingState = NrRlcUm::WAITING_SI_SF;
    rlc->m_windowSize = 512;
    rlc->m_expectedSeqNumber = nr::SequenceNumber10(200);
    rlc->m_vrUr = 200; // awaited (missing) SN
    rlc->m_vrUx = 200;
    rlc->m_vrUh = 200; // no further buffered SNs -> timer is not rescheduled

    NS_TEST_ASSERT_MSG_EQ((int)rlc->m_reassemblingState,
                          (int)NrRlcUm::WAITING_SI_SF,
                          "precondition: reassembler holds a partial SDU");
    NS_TEST_ASSERT_MSG_EQ((rlc->m_keepS0 != nullptr), true, "precondition: S0 is held");

    // t-Reordering expiry: the missing continuation will never arrive.
    rlc->ExpireReorderingTimer();

    NS_TEST_ASSERT_MSG_EQ((int)rlc->m_reassemblingState,
                          (int)NrRlcUm::WAITING_S0_FULL,
                          "stranded partial SDU must reset the reassembly state on t-Reordering "
                          "expiry");
    NS_TEST_ASSERT_MSG_EQ((rlc->m_keepS0 == nullptr),
                          true,
                          "stranded partial SDU (S0) must be discarded on t-Reordering expiry");
    NS_TEST_ASSERT_MSG_EQ(rxPdcp->GetDataReceived().size(),
                          0,
                          "no corrupt SDU may be delivered to PDCP when the partial SDU is "
                          "discarded");

    // Flush the event scheduled by the NrTestPdcp constructor, so that it does not
    // outlive this test case and fire on a destroyed object in the next one.
    Simulator::Destroy();
}

/**
 * @ingroup tests
 *
 * @brief Regression test for the PDCP discard path in NrRlcUm::DoTransmitPdcpPdu.
 *
 * When PDCP discarding is enabled and the head-of-line delay exceeds the discard
 * timer, the incoming RLC SDU must be dropped (TxDrop trace fired) and, crucially,
 * NOT stored in the transmission buffer. Before the fix the code fell through and
 * still enqueued the "discarded" packet, making the discard a no-op and
 * double-counting the drop statistics. This test asserts the packet is not stored.
 */
class NrRlcUmTxPdcpDiscardTestCase : public TestCase
{
  public:
    NrRlcUmTxPdcpDiscardTestCase()
        : TestCase("Test RLC UM TX: PDCP-discarded SDU is not enqueued in the Tx buffer")
    {
    }

    /// Count of TxDrop trace invocations.
    uint32_t m_dropCount{0};

    /**
     * TxDrop trace sink.
     *
     * @param p the dropped packet
     */
    void HandleTxDrop(Ptr<const Packet> p)
    {
        m_dropCount++;
    }

  private:
    void DoRun() override;
};

void
NrRlcUmTxPdcpDiscardTestCase::DoRun()
{
    Ptr<NrRlcUm> rlc = CreateObject<NrRlcUm>();
    rlc->SetAttribute("EnablePdcpDiscarding", BooleanValue(true));
    rlc->SetAttribute("DiscardTimerMs", UintegerValue(10));

    Ptr<NrTestMac> mac = CreateObject<NrTestMac>();
    mac->SetRlcHeaderType(NrTestMac::UM_RLC_HEADER);
    rlc->SetNrMacSapProvider(mac->GetNrMacSapProvider());
    mac->SetNrMacSapUser(rlc->GetNrMacSapUser());
    rlc->TraceConnectWithoutContext(
        "TxDrop",
        MakeCallback(&NrRlcUmTxPdcpDiscardTestCase::HandleTxDrop, this));

    // t = 0 ms: first SDU is stored (Tx buffer is empty, so head-of-line delay is 0).
    Simulator::Schedule(MilliSeconds(0), [rlc]() { rlc->DoTransmitPdcpPdu(Create<Packet>(100)); });

    // t = 20 ms: head-of-line delay (20 ms) exceeds the 10 ms discard timer, so the
    // second SDU must be discarded and must NOT be added to the Tx buffer.
    Simulator::Schedule(MilliSeconds(20), [rlc]() { rlc->DoTransmitPdcpPdu(Create<Packet>(200)); });

    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(rlc->m_txBuffer.size(),
                          1,
                          "the discarded SDU must not be enqueued in the Tx buffer");
    NS_TEST_ASSERT_MSG_EQ(rlc->m_txBufferSize,
                          100,
                          "the Tx buffer size must only account for the stored (first) SDU");
    NS_TEST_ASSERT_MSG_EQ(m_dropCount,
                          1,
                          "the discarded SDU must fire the TxDrop trace exactly once");

    Simulator::Destroy();
}

/**
 * @ingroup tests
 *
 * @brief Regression test for NrRlcUm::ReassembleSnInterval with a gap in the buffer.
 *
 * ReassembleSnInterval iterates over an SN interval that, by design, may contain
 * missing SNs (holes left by losses). Before the fix, the two NS_LOG_LOGIC lines
 * printing it->first / it->second were placed BEFORE the it != end() check,
 * dereferencing m_rxBuffer.end() for every missing SN (undefined behaviour, tripping
 * asserts in debug builds when logging is enabled). This test enables NrRlcUm logging
 * and reassembles an interval with a hole to ensure gaps are skipped safely while the
 * present SNs are reassembled and removed from the reception buffer.
 */
class NrRlcUmReassembleGapTestCase : public TestCase
{
  public:
    NrRlcUmReassembleGapTestCase()
        : TestCase("Test RLC UM RX: ReassembleSnInterval safely skips missing SNs")
    {
    }

  private:
    void DoRun() override;
};

void
NrRlcUmReassembleGapTestCase::DoRun()
{
    // Force the code path that used to dereference m_rxBuffer.end(): the NS_LOG_LOGIC
    // lines only execute (and only crash) when the log component is enabled.
    LogComponentEnable("NrRlcUm", (LogLevel)(LOG_LEVEL_ALL | LOG_PREFIX_ALL));

    Ptr<NrRlcUm> rlc = CreateObject<NrRlcUm>();
    Ptr<NrTestPdcp> rxPdcp = CreateObject<NrTestPdcp>();
    rlc->SetNrRlcSapUser(rxPdcp->GetNrRlcSapUser());

    rlc->m_windowSize = 512;
    rlc->m_reassemblingState = NrRlcUm::WAITING_S0_FULL;

    // Build two self-contained full-SDU PDUs at SN=5 and SN=7, leaving SN=6 as a gap.
    auto makeFullSdu = [](uint16_t sn) {
        Ptr<Packet> pdu = Create<Packet>(20);
        NrRlcHeader h;
        h.SetSequenceNumber(nr::SequenceNumber10(sn));
        h.PushExtensionBit(NrRlcHeader::DATA_FIELD_FOLLOWS);
        h.SetFramingInfo(NrRlcHeader::FIRST_BYTE | NrRlcHeader::LAST_BYTE);
        pdu->AddHeader(h);
        return pdu;
    };
    rlc->m_rxBuffer[5] = makeFullSdu(5);
    rlc->m_rxBuffer[7] = makeFullSdu(7);

    // Reassemble the interval [5, 8): SN=6 is missing and must be skipped without any
    // dereference of m_rxBuffer.end().
    rlc->ReassembleSnInterval(nr::SequenceNumber10(5), nr::SequenceNumber10(8));

    NS_TEST_ASSERT_MSG_EQ(rlc->m_rxBuffer.count(5),
                          0,
                          "SN=5 must be reassembled and removed from the reception buffer");
    NS_TEST_ASSERT_MSG_EQ(rlc->m_rxBuffer.count(7),
                          0,
                          "SN=7 must be reassembled and removed from the reception buffer");
    // NrTestPdcp overwrites its buffer on each delivery, so it holds the last SDU only;
    // a non-empty 20-byte payload confirms a full SDU was reassembled and delivered.
    NS_TEST_ASSERT_MSG_EQ(rxPdcp->GetDataReceived().size(),
                          20,
                          "a full SDU must be reassembled and delivered to PDCP");

    // Flush the event scheduled by the NrTestPdcp constructor, so that it does not
    // outlive this test case and fire on a destroyed object in the next one.
    Simulator::Destroy();
}

class NrRlcUmTestSuite : public TestSuite
{
  public:
    NrRlcUmTestSuite()
        : TestSuite("nr-test-rlc-um-rx", Type::SYSTEM)
    {
        AddTestCase(new NrRlcUmTestCase(), Duration::QUICK);
        AddTestCase(new NrRlcUmReorderingDiscardTestCase(), Duration::QUICK);
        AddTestCase(new NrRlcUmTxPdcpDiscardTestCase(), Duration::QUICK);
        AddTestCase(new NrRlcUmReassembleGapTestCase(), Duration::QUICK);
    }
};

static NrRlcUmTestSuite g_nrRlcUmTestSuite;

} // namespace ns3
