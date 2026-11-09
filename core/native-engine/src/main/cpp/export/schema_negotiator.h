#pragma once
#include <cstdint>
#include <vector>
namespace nox::native::exporting { struct SchemaCapabilities{std::uint32_t schemaVersion{1};std::uint32_t abiVersion{2};bool face{true},eyes{true},hands{true},scene{true};}; struct NegotiatedSchema{std::uint32_t version{0};std::vector<std::uint32_t> supportedFeatures;bool compatible{false};}; class SchemaNegotiator final { public: NegotiatedSchema negotiate(const SchemaCapabilities&,const SchemaCapabilities&)const;}; }