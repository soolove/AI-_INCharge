.. Copyright (c) 2026 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
..
.. SPDX-License-Identifier: GPL-2.0-only

.. highlight:: cpp

EPC model
*********
The simulator currently uses a ported version of the core network (EPC) of LENA ns-3 LTE.
The EPC related files were copied and renamed, adding the ``nr-`` prefix.
Similarly, the ported classes/structures/tests received the ``Nr`` prefix.
For example, LTE's ``EpcEnbApplication`` is the counterpart for NR's ``NrEpcGnbApplication``.
For model details see: https://www.nsnam.org/docs/models/html/lte-design.html#epc-model

The 4G EPC architecture (current) includes the S1 interface, S1 Application Protocol in
the control plane, the S1 User Plane (GTP-U tunneling), the X2 interface for inter-eNB
communication. These would need to be significantly revised for a 5G standalone (SA)
Core (5GC), including moving to a SIP-based 5GC NAS, replacement of S1-AP with the NGAP
protocol, implementation of the UPF, the N4 interface (PFCP protocol), the inter-gNB
Xn interface (NG-RAN), and other concepts around Service-Based Interface (SBI). Almost
the entire EPC model will need to be replaced to model a 5G SA Core

QoS configuration
=================

5G QoS relies on the concepts of QoS rules, QoS flows, and QoS profiles, as
standardized in [TS24501]_. The simulator does not implement all of the
standardized features of 5G QoS, but provides the following capabilities to
allow users to characterize some desired or required QoS features and to
have the simulator models react to this configuration.

The general QoS architecture in 5G, from the perspective of a UE,
is as follows:

  .. code-block:: text

        ┌─────────────────────────────────────────────────────┐
        │                  IP packets                         │
        └────────────────────┬────────────────────────────────┘
                             │
                             ▼
        ┌─────────────────────────────────────────────────────┐
        │                 SDAP Layer                          │
        │                                                     │
        │   • QoS Rule processing                             │
        │   • (possibly many-to-1 mapping)                    │
        │                     │                               │
        │                     ▼                               │
        │  ┌──────────┬──────────┬──────────┬──────────┐      │
        │  │  QoS     │  QoS     │  QoS     │  QoS     │      │
        │  │ Flow 1   │ Flow 2   │ Flow 3   │ Flow N   │      │
        │  └────┬─────┴────┬─────┴────┬─────┴────┬─────┘      │
        │       │          │          │          │            │
        │       └──────┬───┴──┬───────┘  ────┬───┘            │
        │       │                        │                    │
        │       ▼   (many-to-1 mapping)  ▼                    │
        └────────────────────┬────────────────────────────────┘
               │ Data Radio              │Data Radio
               │ Bearer 1                │Bearer K
               │                         │
        ┌──────▼─────────────────────────▼────────┐
        │              PDCP Sublayer              │
        │                                         │
        │  PDCP Entity 1      ...   PDCP Entity K │
        └──────┬─────────────────────────┬────────┘
               │                         │
               │  (1-to-1 mapping)       │
               │ RLC channel 1           │RLC channel K
        ┌──────▼─────────────────────────▼────────┐
        │        RLC Sublayer                     │
        │                                         │
        │  RLC Channel 1      ...   RLC Channel K │
        │  (with queue)              (with queue) │
        └──────┬─────────────────────────┬────────┘
               │Logical  (1-to-1 mapping)│Logical
               │Channel 1                │Channel K
        ┌──────▼─────────────────────────▼────────┐
        │        MAC Layer                        │
        │                                         │
        └──────┬─────────────────────────┬────────┘


The SDAP layer is responsible for mapping IP packets to data radio bearers
based on initially mapping the packets to QoS flows (based on QoS rules),
and then mapping the QoS flows onto data radio bearers.  Each of these
steps is a many-to-one mapping. Once packets are mapped to a data
radio bearer, they are handled by entities at the PDCP layer and RLC
layer; one data radio bearer at the PDCP layer maps to a single RLC
channel at the RLC sublayer, which, in turn, maps to a single logical
channel at the MAC layer.

The overall goals of the simulator configuration are as follows:

1.  QoS configuration should be optional for user-level programs. Default
    configurations should create working data paths. In particular, there
    should be a default data radio bearer that can handle all packets
    in the absence of other configuration.
2.  Users should be able to define additional data radio bearers and
    to create packet filters (based on IP addresses, protocol numbers,
    and port numbers) that will map packets to non-default data radio
    bearers, for the purpose of different handling by the MAC layer
    (e.g., scheduling, HARQ behavior, drop policy).

In this simulation model, to simplify the model, the two-stage,
many-to-one mapping between QoS rules and data radio bearers is
reduced to a one-to-one mapping.  Users can configure multiple QoS
rules but each QoS rule maps to a single QoS flow, which, in turn,
maps to a single data radio bearer.

  .. code-block:: text

        ┌─────────────────────────────────────────────────────┐
        │                  IP Packets                         │
        └────────────────────┬────────────────────────────────┘
                             │
                             ▼
        ┌─────────────────────────────────────────────────────┐
        │                 SDAP Layer                          │
        │                                                     │
        │   • QoS Rule Processing                             │
        │                     │                               │
        │                     ▼                               │
        │  ┌──────────┬              ┬──────────┐             │
        │  │  QoS     │              │  QoS     │             │
        │  │ Flow 1   │              │ Flow K   │             │
        │  └────┬─────┴              ┴───┬-─────┘             │
        │       │                        │                    │
        │       |                        |                    │
        │       │                        │                    │
        │       ▼    (1 to-1 mapping)    ▼                    │
        └────────────────────┬────────────────────────────────┘
               │ Data Radio              │Data Radio
               │ Bearer 1                │Bearer K
               │                         │

The simulation objects that implement this are as follows:

* **NrQosRule**:  The QoS rule is a structure containing a packet filter set,
  a precedence value, and a QoS Flow Identifier (QFI).

* **NrQosRuleClassifier**:  This object holds the set of NrQoSRules and
  is able to classify an IP packet by iterating the rules in order of
  precedence until a match is found.

* **NrQosFlow**: This object holds a QoS Flow Identifier (QFI) and also most
  of the QoS parameters (the QoS profile) such as the 5QI index value for
  pre-defined application QoS profiles, priority, packet delay budget,
  packet error rate, resource type (e.g., guaranteed bit rate).

Additionally, there are properties of the data radio bearers that are
outside of the QoS configuration but that must be configured and coordinated.
For on-network operation, this includes, for example, which mode of RLC
(RLC-UM, RLC-AM) and the number of MAC retransmissions to configure, and for
sidelink operation, additional configuration such as cast type, type of
grant (dynamic or semi-persistent), and the resource retransmission interval.
For on-network operation, the gNB configures the type of bearers and the
RRC ensures that QoS flows are mapped to the right bearers.  For sidelink,
some additional configuration and coordination are required.  In the
simulator, this information is associated with a separate **SidelinkInfo**
parameter that is attached to the QoS rule.  Likewise, the QoS profile
information available in the NrQosFlow must be made available to the lower
layers so that the QoS requirements are met.

5G has the concept of a **PDU Session** that is roughly analogous to a 4G EPS
bearer. Configuration at the moment is done on the basis of QoS Flow, but
could be done on the basis of PDU Session in the future. However, this probably
should be coordinated with any EPC to 5GC upgrade (see above) and NAS upgrade.
Some kind of PDU Session container for multiple QoS flows, and metadata such as
PDU Session ID, DNN (Data Network Name), and S-NSSAI (network slice indicator)
would be needed. At the user-level, the main question would be whether to
maintain backward compatibility with a flow-centric API or refactor it to be
(PDU) session-centric and manage the mappings from PDU Session to QoS Flows to
Data Radio Bearers.

QoS and Bearer Identifiers
==========================

This section documents the mapping and relationships between various identifiers
used throughout the ns-3 5G NR model for QoS flow setup and data bearer
configuration. These identifiers span multiple network layers and domains:

- **NAS Layer**: QoS Flow Identifier (QFI), E-RAB ID
- **RAN/RRC Layer**: Data Radio Bearer ID (DRBID), Logical Channel ID (LCID)
- **MAC Layer**: Logical Channel ID (LCID)
- **GTP-U Layer**: Tunnel Endpoint Identifiers (TEID)

Identifier Mapping
##################

The following table documents the key identifiers and their relationships:

.. table:: QoS and Bearer Identifier Mapping in ns-3 NR

  +------------------+------------------+-------------------------------+
  | Identifier       | Description      | Assignment/Derivation         |
  +==================+==================+===============================+
  | QFI              | QoS Flow         | Assigned by MME; QFI=1 for    |
  |                  | Identifier       | default, QFI 2 reserved for   |
  |                  | (NAS Layer)      | sidelink; subsequent flows    |
  |                  |                  | assigned 3, 4, 5...           |
  +------------------+------------------+-------------------------------+
  | E-RAB ID         | EPC Layer        | Assigned by MME; in this      |
  |                  | Identifier       | model, E-RAB ID = QFI         |
  |                  |                  |                               |
  +------------------+------------------+-------------------------------+
  | DRBID            | Data Radio       | Derived from QFI using        |
  |                  | Bearer ID        | formula: DRBID = QFI + 2;     |
  |                  | (RAN/RRC Layer)  | DRBID 1, 2, 4 reserved        |
  |                  |                  | (SRB1, SRB2, sidelink)        |
  +------------------+------------------+-------------------------------+
  | LCID             | Logical Channel  | Derived from DRBID using      |
  |                  | Identifier       | formula: LCID = DRBID         |
  |                  | (MAC Layer)      | (direct 1:1 mapping);         |
  |                  |                  | LCID 1, 2, 4 reserved         |
  +------------------+------------------+-------------------------------+
  | S1-U SGW TEID    | Tunnel Endpoint  | Assigned by SGW (independent  |
  |                  | Identifier for   | value, not derived from       |
  |                  | S1-U Interface   | QFI/DRBID/LCID); propagated   |
  |                  | (EPC/SGW)        | to MME and gNB for data       |
  |                  |                  | tunneling. The same per-flow  |
  |                  |                  | value is used on S5-U (single |
  |                  |                  | TEID end-to-end)              |
  +------------------+------------------+-------------------------------+
  | S5-U PGW TEID    | Tunnel Endpoint  | Not independently assigned:   |
  |                  | Identifier for   | the PGW echoes the SGW's      |
  |                  | S5-U Interface   | S5-U TEID back in the Create  |
  |                  | (EPC/PGW)        | Session Response when binding |
  |                  |                  | the flow                      |
  +------------------+------------------+-------------------------------+

Key Design Relationships
########################

The model uses the following direct relationships between identifiers:

* **QFI Numbering**: QFI=1 for the default data radio bearer. Subsequent QoS
  flows are assigned QFI 3, 4, 5, ... (QFI 2 is reserved for sidelink and
  is skipped). This numbering is assigned by the MME during flow activation.

* **DRBID Derivation**: DRBID = QFI + 2. Therefore, the default bearer
  (QFI=1) uses DRBID=3. Dedicated bearers use DRBID=5, 6, 7, ... DRBID
  values 1, 2, and 4 are reserved for Signaling Radio Bearers (SRB) and
  sidelink operations.

* **LCID Direct Mapping**: LCID = DRBID. This is a direct 1:1 mapping, not
  an offset. The default data radio bearer uses LCID=3, and dedicated bearers
  use LCID=5, 6, 7, ... LCID values 0-2 are reserved for SRBs, and LCID 4
  is reserved for sidelink.

* **E-RAB ID Alignment**: In this ns-3 model, E-RAB ID = QFI. This ensures
  that the identifier assigned by the MME (E-RAB ID) matches the QFI used at
  the NAS layer, simplifying the implementation while maintaining semantic
  correctness.

* **TEID Independence**: The per-flow TEID is allocated only by the SGW (from
  a counter) and is not derived from QFI, DRBID, or LCID. The same value is
  used end-to-end on both S1-U and S5-U (the SGW performs no data-plane TEID
  translation), and the PGW echoes it back rather than allocating its own.
  These TEIDs identify the tunnel endpoints for GTP-U data transfer across
  the EPC interfaces.

It is important to note that in the 5G NR QoS architecture, there can be
a many-to-one mapping between QFIs and DRBs.  In the ns-3 model at present,
there is a one-to-one mapping, which allows the models to derive DRBID
directly from QFI rather than maintain a separate mapping.

Implementation Details
######################

The identifier assignment occurs across the following code paths:

1. **NAS Layer (UE)**: The UE's NAS entity (NrEpcUeNas) receives QoS flows
   to be activated and queues them. When transitioning to ACTIVE state, it
   installs each flow's rule into the uplink classifier using the QFI already
   set on the rule (the numbering itself is performed by the MME; see below).

2. **MME**: The MME's AddFlow() method in NrEpcMmeApplication assigns E-RAB IDs
   that align with QFI values, using the same numbering scheme: 1 for default,
   then 3, 4, 5... (skipping 2). The MME receives TEID values from SGW and PGW
   and stores them for use in routing and tunneling operations.

3. **RAN/RRC Layer (gNB)**: The gNB's RRC entity (NrUeManager) receives
   activation requests for QoS flows and derives DRBID values using the
   formula DRBID = QFI + 2. The LCID is set equal to the DRBID.

4. **EPC/SGW and PGW**: The SGW allocates one TEID per flow (used on both
   S1-U and S5-U), and the PGW echoes that TEID back when binding the flow.
   These values are exchanged through EPC signaling (CreateSession messages)
   so that the gNB and MME know the correct tunnel endpoints for data
   forwarding.

Control-plane procedures
========================
The following sequence diagrams show the end-to-end control-plane message exchanges as modelled,
across the UE, gNB, MME, SGW and PGW. Two modelling conventions matter when reading them: the S1-AP
primitives are direct in-process SAP calls between ``NrEpcGnbApplication`` and ``NrEpcMmeApplication``
(they are not serialized), whereas the GTP-C (S11/S5) and X2-AP messages are real UDP packets; and
there is no ``InitialContextSetupResponse`` and no explicit NAS PDU on the wire, since the NAS is
modelled as the ``NrEpcUeNas`` state machine (see the :doc:`NAS <nas-layer>` section).

Initial attach and QoS-flow activation
######################################
:numref:`fig-epc-attach` shows the initial UE attach. After RRC connection setup the gNB sends an
S1-AP Initial UE Message to the MME, which creates the session over S11/S5 (one flow context per
default and dedicated QoS flow, carried in a single Create Session Request). The MME then issues the
S1-AP Initial Context Setup Request carrying the E-RAB list (QFI, ``NrQosFlow`` and S1-U TEID); the
gNB sets up the data radio bearers and reconfigures the UE, after which the UE NAS reaches ``ACTIVE``
and installs its uplink ``NrQosRule`` classifier.

.. _fig-epc-attach:

.. figure:: figures/epc/attach.*
   :align: center

   Initial attach and default/dedicated QoS-flow (bearer) activation.

X2-based handover
#################
:numref:`fig-epc-handover` gives the EPC control-plane view of an X2-based handover: X2 preparation
(Handover Request / Acknowledge), SN status transfer and X2-U data forwarding, then, once the UE has
accessed the target, the S1-AP Path Switch Request that triggers the GTP-C Modify Flow exchange
(S11/S5) to move the downlink path to the target gNB, and finally the X2 UE Context Release that frees
the source. The radio-side view of the same procedure is in the :doc:`RRC <rrc-layer>` section.

.. _fig-epc-handover:

.. figure:: figures/epc/handover.*
   :align: center

   EPC control-plane view of an X2-based handover (path switch).

Connection release and teardown
###############################
:numref:`fig-epc-release` shows a full connection release. The RRC Connection Release moves the UE NAS
to ``OFF``; the gNB then sends an S1-AP E-RAB Release Indication per active flow, which drives the
GTP-C Delete Flow chain (S11/S5). The local UE-context removal at the gNB (RRC/MAC/PHY state and the
S1-U TEIDs) runs in the same call stack, in parallel with the GTP-C deletion.

.. _fig-epc-release:

.. figure:: figures/epc/release.*
   :align: center

   Connection release and UE-context teardown.

EPC interfaces
==============
The EPC / core-network interfaces (S1, S5, S11 and X2) are documented in the dedicated
:doc:`Interfaces <s1-interface>` chapter: see the :doc:`S1 <s1-interface>`, :doc:`S5 <s5-interface>`,
:doc:`S11 <s11-interface>` and :doc:`X2 <x2-interface>` interface sections. The identifier mapping
(QFI, E-RAB ID, DRBID, LCID) and TEID-independence relationships described above are referenced from
those pages rather than repeated.
